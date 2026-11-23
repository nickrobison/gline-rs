//! Ready-for-inference tensors for the static-capacity RELEX export.
//!
//! The stock GLiNER ONNX export bakes the relation outputs into constants
//! (the pair-building step is data-dependent, so the tracer froze it). The
//! corrected export therefore takes the entity slots as *inputs*:
//! `span_idx` `[batch, slots, 2]` (word-level, inclusive) and `span_mask`
//! `[batch, slots]`, alongside the usual token-mode tensors. The graph scores
//! every ordered pair of distinct slots, so the caller decides which slots
//! are backed by a decoded entity through `span_mask`.

use super::super::super::pipeline::context::EntityContext;
use super::super::encoded::EncodedInput;
use crate::util::result::Result;
use ndarray::{Array2, Array3};
use ort::session::SessionInputs;

const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTENTION_MASK: &str = "attention_mask";
const TENSOR_WORD_MASK: &str = "words_mask";
const TENSOR_TEXT_LENGTHS: &str = "text_lengths";
pub const TENSOR_SPAN_IDX: &str = "span_idx";
pub const TENSOR_SPAN_MASK: &str = "span_mask";

/// Number of entity slots the static RELEX graph reserves per sequence.
///
/// The relation head scores every ordered pair of distinct slots, so its cost
/// grows quadratically with this constant; 24 slots (552 pairs) is negligible
/// next to the encoder while leaving room for entity-dense documents. Entities
/// beyond this cap take part in entity extraction but not in relations.
pub const RELEX_MAX_SLOTS: usize = 24;

/// Entity slots fed to the static RELEX graph.
pub struct SpanSlots {
    /// `[batch, slots, 2]` word-level `[start, end]` (inclusive) per slot.
    /// Slots without an entity behind them MUST be zero: the graph gathers
    /// word embeddings at these indices regardless of the mask.
    pub idx: Array3<i64>,
    /// `[batch, slots]` 1 for a slot backed by a decoded entity, else 0.
    pub mask: Array2<i64>,
}

impl SpanSlots {
    /// All-invalid slots (used for the entity pass, whose logits do not
    /// depend on the slots at all).
    pub fn zero(batch: usize) -> Self {
        Self {
            idx: Array3::zeros((batch, RELEX_MAX_SLOTS, 2)),
            mask: Array2::zeros((batch, RELEX_MAX_SLOTS)),
        }
    }

    /// Slots backed by the given word-level spans, one list per sequence.
    ///
    /// Slot `i` always corresponds to span `i`: an entry of `None` leaves its
    /// slot invalid (masked out) rather than shifting the following spans down,
    /// so a slot index in `rel_idx` can never be attributed to the wrong span.
    /// Sequences with more entries than [`RELEX_MAX_SLOTS`] keep their first
    /// slots (the remainder simply cannot form relations).
    pub fn from_word_spans(spans: &[Vec<Option<(usize, usize)>>]) -> Self {
        let batch = spans.len();
        let mut slots = Self::zero(batch);
        for (sequence, words) in spans.iter().enumerate() {
            for (slot, word) in words.iter().take(RELEX_MAX_SLOTS).enumerate() {
                if let Some(&(start, end)) = word.as_ref() {
                    slots.idx[[sequence, slot, 0]] = start as i64;
                    slots.idx[[sequence, slot, 1]] = end as i64;
                    slots.mask[[sequence, slot]] = 1;
                }
            }
        }
        slots
    }
}

/// Ready-for-inference tensors (static relex: token inputs + entity slots)
pub struct RelexTensors<'a> {
    pub tensors: SessionInputs<'a, 'a>,
    pub context: EntityContext,
}

impl RelexTensors<'_> {
    pub fn from(encoded: EncodedInput, slots: SpanSlots) -> Result<Self> {
        if slots.idx.shape()[0] != encoded.texts.len() {
            return Err(format!(
                "span slots batch ({}) does not match the input batch ({})",
                slots.idx.shape()[0],
                encoded.texts.len()
            )
            .into());
        }
        let input_ids = ort::value::Tensor::from_array(encoded.input_ids)?;
        let attention_mask = ort::value::Tensor::from_array(encoded.attention_masks)?;
        let word_mask = ort::value::Tensor::from_array(encoded.word_masks)?;
        let text_lengths = ort::value::Tensor::from_array(encoded.text_lengths)?;
        let span_idx = ort::value::Tensor::from_array(slots.idx)?;
        let span_mask = ort::value::Tensor::from_array(slots.mask)?;
        let inputs: Vec<_> = ort::inputs![
            TENSOR_INPUT_IDS => input_ids,
            TENSOR_ATTENTION_MASK => attention_mask,
            TENSOR_WORD_MASK => word_mask,
            TENSOR_TEXT_LENGTHS => text_lengths,
            TENSOR_SPAN_IDX => span_idx,
            TENSOR_SPAN_MASK => span_mask,
        ];
        Ok(Self {
            tensors: inputs.into(),
            context: EntityContext {
                texts: encoded.texts,
                tokens: encoded.tokens,
                entities: encoded.entities,
                relations: encoded.relations,
                num_words: encoded.num_words,
            },
        })
    }

    pub fn inputs() -> [&'static str; 6] {
        [
            TENSOR_INPUT_IDS,
            TENSOR_ATTENTION_MASK,
            TENSOR_WORD_MASK,
            TENSOR_TEXT_LENGTHS,
            TENSOR_SPAN_IDX,
            TENSOR_SPAN_MASK,
        ]
    }
}
