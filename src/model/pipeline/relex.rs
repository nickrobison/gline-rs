//! Pre-defined pipeline for the UniEncoderSpanRelex (token mode + native relations).

use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::{input, output, params};
use super::context::EntityContext;
use crate::util::result::Result;
use composable::*;
use orp::{params::RuntimeParameters, pipeline::*};
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
