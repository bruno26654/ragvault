//! Property tests for the bounded top-k selector.

use proptest::prelude::*;
use ragvault_vector::TopK;

proptest! {
    /// TopK always matches a full sort + truncate reference (NaN dropped),
    /// stays sorted, and never returns duplicates.
    #[test]
    fn matches_full_sort(scores in prop::collection::vec(-1e6f32..1e6, 0..300), k in 0usize..40) {
        let mut topk = TopK::new(k);
        for (id, &s) in scores.iter().enumerate() {
            topk.push(id as u32, s);
        }
        let got = topk.into_sorted();

        let mut reference: Vec<(u32, f32)> = scores
            .iter()
            .enumerate()
            .map(|(id, &s)| (id as u32, s))
            .collect();
        reference.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        reference.truncate(k);
        prop_assert_eq!(got.clone(), reference);

        for w in got.windows(2) {
            prop_assert!(w[0].1 >= w[1].1);
            prop_assert!(w[0].0 != w[1].0);
        }
    }

    /// Merging split streams equals one stream.
    #[test]
    fn merge_equivalence(scores in prop::collection::vec(-1e3f32..1e3, 0..200), k in 1usize..20, split in 0usize..200) {
        let split = split.min(scores.len());
        let mut a = TopK::new(k);
        let mut b = TopK::new(k);
        let mut single = TopK::new(k);
        for (id, &s) in scores.iter().enumerate() {
            if id < split { a.push(id as u32, s); } else { b.push(id as u32, s); }
            single.push(id as u32, s);
        }
        a.merge(b);
        prop_assert_eq!(a.into_sorted(), single.into_sorted());
    }
}
