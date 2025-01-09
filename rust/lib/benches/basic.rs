use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use anyhow::{bail, ensure, Result};
use magika::{ContentType, Features, FeaturesOrRuled, FileType, RuledType, Session, TypeInfo};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;

use ndarray::Array2;

fn build_session_with_core_ml(thread: usize) -> Result<Session> {
    ort::init()
        .with_telemetry(false)
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;
    let mut magika = Session::builder();
    // Apparently, SetIntraOpNumThreads must be called on MacOS, otherwise we get the following
    // error: intra op thread pool must have at least one thread for RunAsync.
    magika = magika
        .with_intra_threads(thread)
        .with_inter_threads(thread)
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .with_parallel_execution(true);
    Ok(magika.build()?)
}

fn identify(session: &mut Session, features: Array2<i32>) {
    session.run_from_input(features).unwrap();
}

pub fn bench_core_ml(c: &mut Criterion) {
    let mut group = c.benchmark_group("magika_with_coreml");
    for thread in 1..9 {
        let mut magika = build_session_with_core_ml(thread).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(thread), &thread, |b, &thread| {
            let features = magika.extract_features(&b"#!/bin/sh\necho hello"[..]).unwrap();
            let input = magika.features(&[features]);

            // we are also measuring the clone :C
            b.iter(|| identify(black_box(&mut magika), black_box(input.clone())));
        });
    }
}

criterion_group!(benches, bench_core_ml);
criterion_main!(benches);
