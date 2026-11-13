// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The Python behavior that decisions prompts and answers depend on.
//!
//! A request's state, criteria and instructions reach the prompt through
//! Python's `str()`, `repr()` and `json.dumps`, and answers through Python's
//! float `sum()`. The canvas noise of a diffusion read comes from Python's
//! `random.Random`. These helpers reproduce them so both frontends send the
//! model the same prompt and return the same numbers.

use std::fmt::Write as _;

use serde_json::{Number, Value};
use sha2::{Digest, Sha512};

/// Python's `repr()` of a `str`.
pub(super) fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if is_printable(ch) => out.push(ch),
            ch => {
                let code = u32::from(ch);
                let _ = match code {
                    0..=0xff => write!(out, "\\x{code:02x}"),
                    0x100..=0xffff => write!(out, "\\u{code:04x}"),
                    _ => write!(out, "\\U{code:08x}"),
                };
            }
        }
    }
    out.push(quote);
    out
}

/// An approximation of Python's `str.isprintable()` for one character: every
/// separator but the space, and every control and format character, is not
/// printable.
fn is_printable(ch: char) -> bool {
    if ch == ' ' {
        return true;
    }
    !(ch.is_control()
        || ch.is_whitespace()
        || matches!(
            ch,
            '\u{ad}'
                | '\u{600}'..='\u{605}'
                | '\u{61c}'
                | '\u{6dd}'
                | '\u{70f}'
                | '\u{180e}'
                | '\u{200b}'..='\u{200f}'
                | '\u{202a}'..='\u{202e}'
                | '\u{2060}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
                | '\u{e000}'..='\u{f8ff}'
        ))
}

/// Python's `repr()` of a JSON number after `json.loads`: an int, or a float.
fn repr_number(number: &Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    repr_float(number.as_f64().unwrap_or(f64::NAN))
}

/// Python's `repr()` of a float: the shortest digits that round-trip, in
/// positional notation for exponents from -4 to 15 and in scientific notation
/// otherwise.
pub(super) fn repr_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    // `{:e}` writes the shortest round-trip digits as `d.ddde<exp>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').expect("`{:e}` writes an exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    // The decimal point sits `point` digits into `digits`.
    let point = exponent + 1;
    let mut out = String::new();
    if value < 0.0 {
        out.push('-');
    }
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exponent = point - 1;
        let sign = if exponent < 0 { '-' } else { '+' };
        let _ = write!(out, "e{sign}{:02}", exponent.unsigned_abs());
    }
    out
}

/// Python's `repr()` of a value after `json.loads`.
pub(super) fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => repr_number(number),
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            let items: Vec<_> = items.iter().map(repr).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Object(map) => {
            let items: Vec<_> = map
                .iter()
                .map(|(key, value)| format!("{}: {}", repr_str(key), repr(value)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Python's `str()` of a value after `json.loads`.
pub(super) fn str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        value => repr(value),
    }
}

/// Python's truthiness of a value after `json.loads`.
pub(super) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Python's `json.dumps(value, ensure_ascii=False)`, or with `indent`,
/// `json.dumps(value, ensure_ascii=False, indent=indent)`.
pub(super) fn json_dumps(value: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_json(&mut out, value, indent, 0);
    out
}

fn write_json(out: &mut String, value: &Value, indent: Option<usize>, depth: usize) {
    let newline = |out: &mut String, depth: usize| {
        if let Some(indent) = indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', indent * depth));
        }
    };
    let separator = if indent.is_some() { "," } else { ", " };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&repr_number(number)),
        Value::String(s) => write_json_str(out, s),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(separator);
                }
                newline(out, depth + 1);
                write_json(out, item, indent, depth + 1);
            }
            newline(out, depth);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(separator);
                }
                newline(out, depth + 1);
                write_json_str(out, key);
                out.push_str(": ");
                write_json(out, item, indent, depth + 1);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

/// A JSON string as Python's `json.dumps(..., ensure_ascii=False)` writes it.
pub(super) fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ch if u32::from(ch) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(ch));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

/// Python's `sum()` of floats, which since Python 3.12 adds with Neumaier's
/// compensated summation.
pub(super) fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter();
    let Some(mut total) = values.next() else {
        return 0.0;
    };
    let mut compensation = 0.0;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// Python's `random.Random`, seeded from bytes, as far as `randrange` needs.
pub(super) struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` with a `bytes` seed.
    pub(super) fn from_bytes(seed: &[u8]) -> Self {
        // Python seeds from `int.from_bytes(seed + sha512(seed).digest())`,
        // big-endian, then splits the integer into 32-bit words from the least
        // significant one.
        let mut bytes = seed.to_vec();
        bytes.extend_from_slice(&Sha512::digest(seed));
        let first = bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len());
        let significant = &bytes[first..];
        let mut key: Vec<u32> = significant
            .rchunks(4)
            .map(|chunk| chunk.iter().fold(0u32, |word, &b| (word << 8) | u32::from(b)))
            .collect();
        if key.is_empty() {
            key.push(0);
        }
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_by_array(&key);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            let prev = self.state[i - 1];
            self.state[i] =
                1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            let prev = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prev = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.index >= 624 {
            for k in 0..624 {
                let y = (self.state[k] & UPPER) | (self.state[(k + 1) % 624] & LOWER);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.state[k] = self.state[(k + 397) % 624] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.randrange(n)` for `0 < n < 2**32`.
    pub(super) fn randrange(&mut self, n: u32) -> u32 {
        assert!(n > 0, "randrange needs a non-empty range");
        let bits = u32::BITS - n.leading_zeros();
        loop {
            let value = self.next_u32() >> (32 - bits);
            if value < n {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn random_matches_python() {
        // `random.Random(hashlib.sha256(json.dumps(ids).encode()).digest())`
        // on Python 3.13.
        let draws = |ids: &str, n: u32| {
            let seed = sha2::Sha256::digest(ids.as_bytes());
            let mut random = PyRandom::from_bytes(&seed);
            (0..5).map(|_| random.randrange(n)).collect::<Vec<_>>()
        };
        assert_eq!(draws("[1, 2, 3]", 1000), [618, 882, 617, 735, 455]);
        assert_eq!(
            draws("[1, 2, 3]", 262_144),
            [233_032, 152_573, 234_541, 59773, 132_719]
        );
        assert_eq!(draws("[]", 7), [0, 3, 5, 0, 2]);
        // `random.Random(b"").getrandbits(32)`.
        let mut empty = PyRandom::from_bytes(b"");
        assert_eq!(
            [empty.next_u32(), empty.next_u32(), empty.next_u32()],
            [4_124_137_760, 1_951_988_028, 2_557_274_880]
        );
    }

    #[test]
    fn float_repr_matches_python() {
        for (value, expected) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123_456.789, "123456.789"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (2.5e100, "2.5e+100"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(repr_float(value), expected, "{value}");
        }
    }

    #[test]
    fn str_and_repr_match_python() {
        assert_eq!(str(&json!("plain")), "plain");
        assert_eq!(str(&json!(null)), "None");
        assert_eq!(str(&json!(true)), "True");
        assert_eq!(str(&json!(3)), "3");
        assert_eq!(str(&json!(2.0)), "2.0");
        assert_eq!(
            str(&json!({"a": [1, "b's", null], "c": {"d": false}})),
            r#"{'a': [1, "b's", None], 'c': {'d': False}}"#
        );
        assert_eq!(repr_str("tab\there"), r"'tab\there'");
        assert_eq!(repr_str("both ' and \""), r#"'both \' and "'"#);
        assert_eq!(repr_str("bell\u{7}"), r"'bell\x07'");
        assert_eq!(repr_str("Français 日本語"), "'Français 日本語'");
    }

    #[test]
    fn json_dumps_matches_python() {
        let value = json!({"message": "Français 日本語", "n": [1, 2.5, null], "q": "a\"b\n\u{1}"});
        assert_eq!(
            json_dumps(&value, None),
            r#"{"message": "Français 日本語", "n": [1, 2.5, null], "q": "a\"b\n\u0001"}"#
        );
        assert_eq!(
            json_dumps(
                &json!({"k": "é", "e": [], "o": {}, "l": [1, {"x": 1}]}),
                Some(2)
            ),
            "{\n  \"k\": \"é\",\n  \"e\": [],\n  \"o\": {},\n  \"l\": [\n    1,\n    {\n      \"x\": 1\n    }\n  ]\n}"
        );
        assert_eq!(json_dumps(&json!([1, 2, 3]), None), "[1, 2, 3]");
    }

    #[test]
    fn sum_compensates_like_python() {
        // `sum([0.1] * 10)` is exactly 1.0 since Python 3.12.
        assert_eq!(sum(std::iter::repeat_n(0.1, 10)), 1.0);
        assert_eq!(sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(sum([0.5]), 0.5);
    }

    #[test]
    fn truthiness_matches_python() {
        for value in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!truthy(&value), "{value}");
        }
        for value in [
            json!(true),
            json!(1),
            json!("no"),
            json!([0]),
            json!({"a": 0}),
        ] {
            assert!(truthy(&value), "{value}");
        }
    }
}
