//! Relex decoding for the `UniEncoderSpanRelex` ONNX export.
//!
//! The shipped `nickrobison/gliner-relex-onnx` model is a UniEncoderSpanRelex
//! export. Its `logits` output is laid out as
//! `[batch_size, sequence_length, num_ent_classes, num_idx_classes]` — the last
//! axis carrying the start/end/inside group — and it additionally exposes native
//! relation outputs (`rel_logits`, `rel_idx`, `rel_mask`).
//!
//! The token-mode decoder shipped with the binding expects the start/end/inside
//! group in the leading axis, i.e. `[3, batch_size, num_words, num_classes]`. This
//! decoder permutes the native tensor into that layout before running the token-mode
//! span logic, and then decodes the native relation outputs on top of the spans.

use super::greedy::GreedySearch;
use super::sort::SpanSort;
use super::token::TensorsToDecoded as TokenTensorsToDecoded;
use super::SpanOutput;
use crate::model::output::tensors::TensorOutput;
use crate::text::span::Span;
use crate::util::math::sigmoid;
use crate::util::result::Result;
use composable::Composable;

const TENSOR_LOGITS: &str = "logits";
const TENSOR_REL_LOGITS: &str = "rel_logits";
const TENSOR_REL_IDX: &str = "rel_idx";
const TENSOR_REL_MASK: &str = "rel_mask";

/// Decoding of the native relation outputs on their own.
///
/// Used as the second pass of the static-capacity RELEX flow: the graph emits
/// `rel_idx` slot indices (`span_idx` positions fed by the caller), and every
/// slot `i < n` corresponds to the `i`-th decoded entity span, so the slot
/// indices double as span-list indices. The caller validates them against the
/// decoded span list.
pub struct RelationsToDecoded {
    relation_threshold: f32,
}

impl RelationsToDecoded {
    pub fn new(relation_threshold: f32) -> Self {
        Self { relation_threshold }
    }

    pub fn outputs() -> [&'static str; 4] {
        [
            TENSOR_LOGITS,
            TENSOR_REL_LOGITS,
            TENSOR_REL_IDX,
            TENSOR_REL_MASK,
        ]
    }

    /// Decode the relation tensors into per-sequence relations.
    ///
    /// `max_spans` caps the slot indices the caller can map back to decoded
    /// spans; pairs referencing a slot beyond it are dropped.
    pub fn decode(
        &self,
        input: &TensorOutput,
        max_spans: &[usize],
    ) -> Result<Vec<Vec<RelexRelation>>> {
        let context = &input.context;
        let batch_size = context.texts.len();

        let rel_logits_arr = input
            .tensors
            .get(TENSOR_REL_LOGITS)
            .ok_or("rel_logits not found in model output")?
            .try_extract_array::<f32>()?;
        let rel_idx_arr = input
            .tensors
            .get(TENSOR_REL_IDX)
            .ok_or("rel_idx not found in model output")?
            .try_extract_array::<i64>()?;
        let rel_mask_arr = input
            .tensors
            .get(TENSOR_REL_MASK)
            .ok_or("rel_mask not found in model output")?
            .try_extract_array::<bool>()?;

        let rel_logits_shape = rel_logits_arr.shape();
        let num_pairs = rel_logits_shape[1];
        let num_rels = context.relations.len();

        let rel_mask_v = rel_mask_arr.slice(ndarray::s![.., ..]);

        let mut relations: Vec<Vec<RelexRelation>> = Vec::with_capacity(batch_size);
        for sequence_id in 0..batch_size {
            let rel_idx_v = rel_idx_arr.slice(ndarray::s![sequence_id, .., ..]);
            let rel_logits_v = rel_logits_arr.slice(ndarray::s![sequence_id, .., ..]);

            let mut sequence_relations: Vec<RelexRelation> = Vec::new();
            for pair in 0..num_pairs {
                let subject = rel_idx_v.get((pair, 0)).copied().unwrap_or(-1);
                let object = rel_idx_v.get((pair, 1)).copied().unwrap_or(-1);
                if !rel_mask_v
                    .get((sequence_id, pair))
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                if subject < 0 || object < 0 {
                    continue;
                }
                let (subject, object) = (subject as usize, object as usize);
                if subject >= max_spans.get(sequence_id).copied().unwrap_or(0)
                    || object >= max_spans.get(sequence_id).copied().unwrap_or(0)
                {
                    continue;
                }
                for rel_id in 0..num_rels {
                    // one logit per configured relation: there is no background
                    // column in `rel_logits`, an unrelated pair simply scores
                    // below the threshold for every class.
                    let prob = sigmoid(rel_logits_v.get((pair, rel_id)).copied().unwrap_or(0.0));
                    if prob > self.relation_threshold {
                        sequence_relations.push(RelexRelation {
                            subject_index: subject,
                            object_index: object,
                            predicate: context.relations[rel_id].clone(),
                            probability: prob,
                        });
                    }
                }
            }
            relations.push(sequence_relations);
        }

        Ok(relations)
    }
}

impl Composable<TensorOutput<'_>, Vec<Vec<RelexRelation>>> for RelationsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<Vec<Vec<RelexRelation>>> {
        // Standalone use (no span list at hand): accept every valid slot.
        let unbounded = vec![usize::MAX; input.context.texts.len()];
        self.decode(&input, &unbounded)
    }
}

/// A relation between two entity spans, as decoded from the native RELEX outputs.
#[derive(Debug, Clone)]
pub struct RelexRelation {
    /// Index of the subject span within the sequence's span list.
    pub subject_index: usize,
    /// Index of the object span within the sequence's span list.
    pub object_index: usize,
    /// Relation class label.
    pub predicate: String,
    /// Confidence of the relation.
    pub probability: f32,
}

/// Output of the relex decoding stage: decoded entity spans plus native relations.
#[derive(Debug, Clone)]
pub struct RelexOutput {
    /// Decoded entity spans, one list of spans per input sequence.
    pub spans: SpanOutput,
    /// Decoded relations, one list of relations per input sequence.
    pub relations: Vec<Vec<RelexRelation>>,
}

impl RelexOutput {
    pub fn new(spans: SpanOutput, relations: Vec<Vec<RelexRelation>>) -> Self {
        Self { spans, relations }
    }
}

/// Decoding method for the UniEncoderSpanRelex (token mode + native relations).
pub struct TensorsToDecoded {
    threshold: f32,
    relation_threshold: f32,
    flat_ner: bool,
    dup_label: bool,
    multi_label: bool,
}

impl TensorsToDecoded {
    pub fn new(
        threshold: f32,
        relation_threshold: f32,
        flat_ner: bool,
        dup_label: bool,
        multi_label: bool,
    ) -> Self {
        Self {
            threshold,
            relation_threshold,
            flat_ner,
            dup_label,
            multi_label,
        }
    }

    pub fn outputs() -> [&'static str; 4] {
        [
            TENSOR_LOGITS,
            TENSOR_REL_LOGITS,
            TENSOR_REL_IDX,
            TENSOR_REL_MASK,
        ]
    }

    fn decode(&self, input: &TensorOutput) -> Result<RelexOutput> {
        let context = &input.context;
        let batch_size = context.texts.len();
        let token_decoder = TokenTensorsToDecoded::new(self.threshold);

        // ------------------------------------------------------------------
        // NER: the native `logits` tensor is [batch, sequence, classes, 3] (the
        // trailing axis is the start/end/inside group) and must be permuted to the
        // token-decoder layout [3, batch, sequence, classes].
        // ------------------------------------------------------------------
        let logits = input
            .tensors
            .get(TENSOR_LOGITS)
            .ok_or("logits not found in model output")?;
        let array = logits.try_extract_array::<f32>()?;
        let native_shape: Vec<i64> = logits.shape().iter().map(|&d| d as i64).collect();
        let _ = native_shape;
        let permuted = array.permuted_axes(vec![3usize, 0, 1, 2]);

        let mut spans: Vec<Vec<Span>> = Vec::with_capacity(batch_size);
        for sequence_id in 0..batch_size {
            // get a slice for the current sequence (2nd dimension of the permuted layout)
            let scores = permuted.slice(ndarray::s![.., sequence_id, .., ..]);
            let scores_start = scores.slice(ndarray::s![0, .., ..]);
            let scores_end = scores.slice(ndarray::s![1, .., ..]);
            let scores_inside = scores.slice(ndarray::s![2, .., ..]);

            let candidates = token_decoder.generate_spans(&scores_start, &scores_end);
            let mut sequence_spans: Vec<Span> = Vec::new();
            for (start_token, end_token, class) in candidates {
                let score = token_decoder
                    .compute_span_score((start_token, end_token, class), &scores_inside);
                if score < self.threshold {
                    continue;
                }
                let span =
                    context.create_span(sequence_id, start_token, end_token, class, score)?;
                sequence_spans.push(span);
            }
            spans.push(sequence_spans);
        }

        // Sort + greedy over the decoded spans, exactly like the token-mode pipeline.
        let sorted_spans = SpanOutput::new(context.texts.clone(), context.entities.clone(), spans);
        let sorted_spans = SpanSort::default().apply(sorted_spans)?;
        let span_output: SpanOutput =
            GreedySearch::new(self.flat_ner, self.dup_label, self.multi_label)
                .apply(sorted_spans)?;

        // ------------------------------------------------------------------
        // Relations from the native RELEX outputs. On the legacy 4-input
        // export these tensors are structurally constant (rel_idx filled with
        // -1, rel_mask all false), so this loop decodes nothing there; on the
        // static-capacity export the single-run decode is not used at all
        // (see `RelexRunner`).
        // ------------------------------------------------------------------
        let max_spans: Vec<usize> = span_output.spans.iter().map(|s| s.len()).collect();
        let relations =
            RelationsToDecoded::new(self.relation_threshold).decode(input, &max_spans)?;

        Ok(RelexOutput {
            spans: span_output,
            relations,
        })
    }
}

impl Composable<TensorOutput<'_>, RelexOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<RelexOutput> {
        self.decode(&input)
    }
}
