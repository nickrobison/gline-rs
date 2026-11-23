//! Pre-defined pipeline for the UniEncoderSpanRelex (token mode + native relations).

use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::{input, output, params};
use super::context::EntityContext;
use crate::util::result::Result;
use composable::*;
use orp::{model::Model, params::RuntimeParameters, pipeline::*};
use std::collections::HashSet;
use std::path::Path;

/// Generic relex token-level pipeline: UniEncoderSpanRelex decoding with native relations.
pub struct RelexTokenPipeline<S, T> {
    splitter: S,
    tokenizer: T,
    expected_inputs: HashSet<&'static str>,
    expected_outputs: HashSet<&'static str>,
}

impl<'a, S: Splitter, T: Tokenizer> Pipeline<'a> for RelexTokenPipeline<S, T> {
    type Input = input::text::TextInput;
    type Output = output::decoded::RelexOutput;
    type Context = EntityContext;
    type Parameters = params::Parameters;

    fn pre_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::token::EncodedToTensors::default(),
            input::tensors::token::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(
        &self,
        params: &Self::Parameters,
    ) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),
            output::decoded::relex::TensorsToDecoded::new(
                params.threshold,
                params.relation_threshold,
                params.flat_ner,
                params.dup_label,
                params.multi_label,
            )
        ]
    }

    fn expected_inputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(self.expected_inputs.iter().copied())
    }

    fn expected_outputs(&self, _params: &Self::Parameters) -> Option<impl Iterator<Item = &str>> {
        Some(self.expected_outputs.iter().copied())
    }
}

/// Specific implementation using HF tokenizer and default splitter
impl RelexTokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
            expected_inputs: input::tensors::token::TokenTensors::inputs()
                .into_iter()
                .collect(),
            expected_outputs: output::decoded::relex::TensorsToDecoded::outputs()
                .into_iter()
                .collect(),
        })
    }
}

/// Shorthand for the default relex token pipeline type (eases disambiguation when calling `GLiNER::new`)
pub type RelexTokenMode =
    RelexTokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer>;

/// Specific GLiNER implementation using the default relex token-mode pipeline
impl super::super::GLiNER<RelexTokenMode> {
    pub fn new<P: AsRef<Path>>(
        params: params::Parameters,
        runtime_params: RuntimeParameters,
        tokenizer_path: P,
        model_path: P,
    ) -> Result<Self> {
        Ok(Self {
            params,
            model: super::super::Model::new(model_path, runtime_params)?,
            pipeline: RelexTokenPipeline::new(tokenizer_path)?,
        })
    }
}

/// Two-pass runner for RELEX extraction.
///
/// The relation head of the original GLiNER ONNX export is unusable: its
/// pair-building step is data-dependent, so the ONNX tracer froze `rel_idx`
/// and `rel_mask` into constants (all `-1` / all `false`) baked to whatever
/// the export's dummy batch decoded. The corrected export therefore takes the
/// entity slots as *inputs* (`span_idx`, `span_mask`) and scores every
/// ordered pair of the fixed slot set.
///
/// This runner drives that graph in two passes over one tokenization:
///
/// 1. entity pass — all slots invalid; entities are decoded from `logits`
///    exactly as in the single-pass pipeline;
/// 2. relation pass — the decoded entities fill the slots, and the native
///    relation outputs are decoded against them.
///
/// Legacy 4-input exports (whose relation outputs are the baked constants
/// described above) are still supported in a degraded single pass: entities
/// decode fine, relations always come back empty.
pub struct RelexRunner<S, T> {
    splitter: S,
    tokenizer: T,
}

impl RelexRunner<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
        })
    }
}

impl<S: Splitter, T: Tokenizer> RelexRunner<S, T> {
    /// Run entity and relation extraction over `input`.
    ///
    /// The model must either be a legacy 4-input relex export (relations
    /// degraded, see above) or a static-capacity 6-input export carrying
    /// `span_idx`/`span_mask`.
    pub fn extract(
        &self,
        model: &Model,
        input: input::text::TextInput,
        params: &params::Parameters,
    ) -> Result<output::decoded::RelexOutput> {
        let input_names = model.input_names();
        let static_graph = input_names
            .iter()
            .any(|name| name == input::tensors::relex::TENSOR_SPAN_IDX);

        // ---- shared pre-processing: tokenize and prompt-encode once ----------
        let tokenized = input::tokenized::RawToTokenized::new(&self.splitter, params.max_length)
            .apply(input)?;
        let prompt = input::prompt::TokenizedToPrompt::default().apply(tokenized)?;
        let encoded = input::encoded::PromptsToEncoded::new(&self.tokenizer).apply(prompt)?;
        let batch = encoded.texts.len();

        if !static_graph {
            // ---- legacy single pass (relations structurally unavailable) ----
            let tensors = input::tensors::token::TokenTensors::from(encoded)?;
            let outputs = model.run_raw(tensors.tensors)?;
            let tensor_output = output::tensors::TensorOutput::from(outputs, tensors.context);
            let decoder = output::decoded::relex::TensorsToDecoded::new(
                params.threshold,
                params.relation_threshold,
                params.flat_ner,
                params.dup_label,
                params.multi_label,
            );
            return decoder.apply(tensor_output);
        }

        // ---- pass 1: entities (slots all invalid) ----------------------------
        let tensors = input::tensors::relex::RelexTensors::from(
            encoded.clone(),
            input::tensors::relex::SpanSlots::zero(batch),
        )?;
        let outputs = model.run_raw(tensors.tensors)?;
        let tensor_output = output::tensors::TensorOutput::from(outputs, tensors.context);
        let decoder = output::decoded::relex::TensorsToDecoded::new(
            params.threshold,
            params.relation_threshold,
            params.flat_ner,
            params.dup_label,
            params.multi_label,
        );
        let entity_output = decoder.apply(tensor_output)?;

        // ---- slots: map decoded spans (char offsets) back to word indices ----
        let word_spans: Vec<Vec<(usize, usize)>> = entity_output
            .spans
            .spans
            .iter()
            .enumerate()
            .map(|(sequence, spans)| {
                let tokens = encoded.tokens.get(sequence).cloned().unwrap_or_default();
                spans
                    .iter()
                    .filter_map(|span| {
                        let (start, end) = span.offsets();
                        let start_word = tokens.iter().position(|t| t.start() == start)?;
                        let end_word = tokens.iter().position(|t| t.end() == end)?;
                        Some((start_word, end_word))
                    })
                    .collect()
            })
            .collect();
        let slots = input::tensors::relex::SpanSlots::from_word_spans(&word_spans);

        // ---- pass 2: relations over the decoded entities ----------------------
        let relations = if encoded.relations.is_empty() {
            vec![Vec::new(); batch]
        } else {
            let tensors = input::tensors::relex::RelexTensors::from(encoded, slots)?;
            let outputs = model.run_raw(tensors.tensors)?;
            let tensor_output = output::tensors::TensorOutput::from(outputs, tensors.context);
            let max_spans: Vec<usize> = entity_output.spans.spans.iter().map(|s| s.len()).collect();
            output::decoded::relex::RelationsToDecoded::new(params.relation_threshold)
                .decode(&tensor_output, &max_spans)?
        };

        Ok(output::decoded::RelexOutput {
            spans: entity_output.spans,
            relations,
        })
    }
}
