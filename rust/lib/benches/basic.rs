use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use anyhow::{bail, ensure, Result};
use magika::{ContentType, Features, FeaturesOrRuled, FileType, RuledType, Session, TypeInfo};
use ort::execution_providers::CoreMLExecutionProvider;

fn build_session_with_core_ml(thread: usize) -> Result<Session> {
    ort::init()
        .with_telemetry(false)
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;
    let mut magika = Session::builder();
    // Apparently, SetIntraOpNumThreads must be called on MacOS, otherwise we get the following
    // error: intra op thread pool must have at least one thread for RunAsync.
    magika = magika.with_intra_threads(thread);
    Ok(magika.build()?)
}

fn identify(session: &mut Session, bytes: &[u8])  {
    session.identify_content_sync(bytes).unwrap();
}

pub fn bench_core_ml(c: &mut Criterion) {
    let mut group = c.benchmark_group("magika_with_coreml");
    for thread in 1..9 {
        let mut magika = build_session_with_core_ml(thread).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(thread), &thread, |b, &thread| {
            b.iter(|| identify(black_box(&mut magika), black_box(&b"#!/bin/sh\necho hello"[..])))
        });
    }

}

criterion_group!(benches, bench_core_ml);
criterion_main!(benches);
