use super::super::super::pipeline::context::EntityContext;
use super::super::encoded::EncodedInput;
use crate::util::result::Result;
use composable::Composable;
use ort::session::SessionInputs;

const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTENTION_MASK: &str = "attention_mask";
const TENSOR_WORD_MASK: &str = "words_mask";
const TENSOR_TEXT_LENGTHS: &str = "text_lengths";

/// Ready-for-inference tensors (token mode)
pub struct TokenTensors<'a> {
    pub tensors: SessionInputs<'a, 'a>,
    pub context: EntityContext,
}

impl TokenTensors<'_> {
    pub fn from(encoded: EncodedInput) -> Result<Self> {
        let input_ids = ort::value::Tensor::from_array(encoded.input_ids)?;
        let attention_mask = ort::value::Tensor::from_array(encoded.attention_masks)?;
        let word_mask = ort::value::Tensor::from_array(encoded.word_masks)?;
        let text_lengths = ort::value::Tensor::from_array(encoded.text_lengths)?;
        let inputs: Vec<_> = ort::inputs![
            TENSOR_INPUT_IDS => input_ids,
            TENSOR_ATTENTION_MASK => attention_mask,
            TENSOR_WORD_MASK => word_mask,
            TENSOR_TEXT_LENGTHS => text_lengths,
        ];
        Ok(Self {
            tensors: inputs.into(),
            context: EntityContext {
                texts: encoded.texts,
                tokens: encoded.tokens,
                entities: encoded.entities,
                num_words: encoded.num_words,
            },
        })
    }

    pub fn inputs() -> [&'static str; 4] {
        [
            TENSOR_INPUT_IDS,
            TENSOR_ATTENTION_MASK,
            TENSOR_WORD_MASK,
            TENSOR_TEXT_LENGTHS,
        ]
    }
}

/// Composable: Encoded => TokenTensors
#[derive(Default)]
pub struct EncodedToTensors {}

impl<'a> Composable<EncodedInput, TokenTensors<'a>> for EncodedToTensors {
    fn apply(&self, input: EncodedInput) -> Result<TokenTensors<'a>> {
        TokenTensors::from(input)
    }
}

/// Composable: TokenTensors => (SessionInput, TensorsMeta)
#[derive(Default)]
pub struct TensorsToSessionInput {}

impl<'a> Composable<TokenTensors<'a>, (SessionInputs<'a, 'a>, EntityContext)>
    for TensorsToSessionInput
{
    fn apply(&self, input: TokenTensors<'a>) -> Result<(SessionInputs<'a, 'a>, EntityContext)> {
        Ok((input.tensors, input.context))
    }
}
