// Chat module for monitoring and responding to text file communication
// Monitors .ai/chats/codriver.txt for new messages

pub mod monitor;
pub mod parser;
pub mod writer;

pub use monitor::ChatMonitor;
pub use parser::{ChatMessage, parse_message};
pub use writer::ChatWriter;
