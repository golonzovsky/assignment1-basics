use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,
    prev: Option<u32>,
    next: Option<u32>,
    len: u32,
}

impl Symbol {
    pub fn merge_with(&mut self, other: &Self, new_c: u32) {
        self.c = new_c;
        self.len += other.len;
        self.next = other.next;
    }
}

#[derive(Clone, Default)]
struct Word {
    symbols: Vec<Symbol>,
}

pub fn train_bpe(
    _input_path: &str,
    _vocab_size: usize,
    _special_tokens: &[String],
) -> Result<(HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), String> {
    // TODO: Implement
    let vocab = HashMap::new();
    let merges = Vec::new();
    
    Ok((vocab, merges))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_bpe() {
        // NOTE: this is just dummy to trigger debugger
        let input_path = "../tests/fixtures/corpus.en";
        let special_tokens = vec!["<|endoftext|>".to_string()];
        
        let result = train_bpe(input_path, 500, &special_tokens);
        assert!(result.is_ok());
        
        let (vocab, merges) = result.unwrap();
        assert!(!vocab.is_empty());
        assert!(!merges.is_empty());

        println!("Vocab size: {}", vocab.len());
        println!("Merges count: {}", merges.len());
    }
}