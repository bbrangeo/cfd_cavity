//! Sondage des limites GPU (taille max des buffers) pour dimensionner les
//! buffers SoA D3Q19 du module uniform3d (19 nœuds × n f32 par population).
//! Usage : cargo run --release --example probe_limits

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("aucun adaptateur GPU");
    let info = adapter.get_info();
    println!("Adaptateur : {} ({:?})", info.name, info.backend);
    let limits = adapter.limits();
    println!(
        "max_buffer_size = {} octets ({:.1} MiB)",
        limits.max_buffer_size,
        limits.max_buffer_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "max_storage_buffer_binding_size = {} octets ({:.1} MiB)",
        limits.max_storage_buffer_binding_size,
        limits.max_storage_buffer_binding_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "max_compute_workgroup_size_z = {} | max_dispatch z = {}",
        limits.max_compute_workgroup_size_z, limits.max_compute_workgroups_per_dimension
    );
    // Buffer 19 nœuds × 7,076,352 (96×384×192) × 4 octets = 538 Mo
    let need: u64 = 19 * 96 * 384 * 192 * 4;
    println!(
        "Besoin V1 (1 population 19 dirs, 96×384×192) = {need} octets ({:.1} MiB) : {}",
        need as f64 / (1024.0 * 1024.0),
        if limits.max_buffer_size >= need { "OK (buffer unique)" } else { "INSUFFISANT (split requis)" }
    );
}
