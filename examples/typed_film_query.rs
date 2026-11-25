//! Verify every supported answer to witnessed FB15k-237 film queries against a set join.
//! See `docs/experiments/typed-film-query/README.md` for data preparation.
use heyting::{answer_query, FuzzyKg, Godel, Query, QueryConfig};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        env::args()
            .nth(1)
            .ok_or("expected verification directory")?,
    );
    let contents = fs::read_to_string(path.join("edges.tsv"))?;
    let expected = fs::read_to_string(path.join("expected.tsv"))?;
    let (queries, answers) = verify(&contents, &expected)?;
    println!("Verified {queries} queries and {answers} supported answers.");
    Ok(())
}

fn verify(
    contents: &str,
    expected_rows: &str,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let mut lines = contents.lines();
    let entities = lines.next().ok_or("missing size")?.parse()?;
    if entities == 0 {
        return Err("empty entity index".into());
    }
    let mut kg = FuzzyKg::new(entities);
    for line in lines {
        let row = parse(line)?;
        let [head, relation, tail] = row[..] else {
            return Err("bad triple".into());
        };
        if head >= entities || tail >= entities {
            return Err("invalid entity index".into());
        }
        kg.add_edge(head, relation, tail, 1.0);
    }
    let (mut queries, mut answers) = (0, 0);
    for line in expected_rows.lines() {
        let row = parse(line)?;
        let [director, profession, expected @ ..] = &row[..] else {
            return Err("missing query anchors".into());
        };
        if *director >= entities || *profession >= entities {
            return Err("invalid query anchor".into());
        }
        let query = Query::intersection(vec![
            Query::anchor(*director, 0).then(1),
            Query::anchor(*profession, 2),
        ]);
        let degrees = answer_query::<Godel>(&kg, &query, &QueryConfig::exact());
        let supported: Vec<_> = degrees
            .iter()
            .enumerate()
            .filter_map(|(i, degree)| (*degree > 0.0).then_some(i))
            .collect();
        if supported != expected {
            return Err(format!("answer mismatch for query {director}, {profession}").into());
        }
        queries += 1;
        answers += supported.len();
    }
    if queries == 0 {
        return Err("no expected queries".into());
    }
    Ok((queries, answers))
}

fn parse(line: &str) -> Result<Vec<usize>, std::num::ParseIntError> {
    line.split_whitespace().map(str::parse).collect()
}

#[cfg(test)]
mod tests {
    use super::verify;

    #[test]
    fn catches_equal_size_wrong_answer() {
        let edges = "5\n0 0 1\n1 1 2\n3 2 2\n3 2 4";
        assert_eq!(verify(edges, "0 3 2").unwrap(), (1, 1));
        let error = verify(edges, "0 3 4").unwrap_err();
        assert!(error.to_string().contains("answer mismatch"));
    }
}
