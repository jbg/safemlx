#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid chat_template: {0}")]
    InvalidChatTemplate(String),

    #[error(
        "chat_template collection has no default template; available templates: {available:?}"
    )]
    AmbiguousChatTemplate { available: Vec<String> },

    #[error(transparent)]
    RenderTemplate(#[from] minijinja::Error),

    /// continue_final_message is set but the final message does not appear in the chat after   
    /// applying the chat template! This can happen if the chat template deletes portions of
    /// the final message. Please verify the chat template and final message in your chat to
    /// ensure they are compatible.
    #[error(
        "continue_final_message is set but the final message does not appear in the chat after applying the chat template!"
    )]
    FinalMsgNotInChat,

    #[error(transparent)]
    Encode(#[from] tokenizers::tokenizer::Error),

    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
}
