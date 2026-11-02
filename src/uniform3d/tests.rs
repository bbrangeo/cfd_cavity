//! Tests unitaires 3D : snapshot b"SNP3" (roundtrip, atomicité),
//! interpolation trilinéaire, re-échelonnage vitesse au restart.

use crate::uniform3d::snapshot;

// ── Snapshot 3D (export/import binaire, format b"SNP3") ──────────────────

#[test]
fn snapshot_3d_roundtrip_is_bitwise_identical() {
    let nx = 10usize;
    let ny = 14usize;
    let nz = 8usize;
    let n = nx * ny * nz;
    let rho: Vec<f64> = (0..n).map(|k| 1.0 + 1e-4 * k as f64).collect();
    let ux: Vec<f64> = (0..n).map(|k| 0.001 * k as f64).collect();
    let uy: Vec<f64> = (0..n).map(|k| -0.002 * k as f64).collect();
    let uz: Vec<f64> = (0..n).map(|k| 0.0005 * k as f64).collect();
    let t: Vec<f64> = (0..n).map(|k| 0.5 - k as f64 / n as f64).collect();

    let dir = std::env::temp_dir().join(format!("cav_snap3d_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snapshot.bin");

    snapshot::write_snapshot_3d_atomic(&path, nx, ny, nz, 1234, 0.02 * (ny - 1) as f64, &rho, &ux, &uy, &uz, &t)
        .expect("écriture atomique");

    let (rnx, rny, rnz, rstep, rus, rrho, rux, ruy, ruz, rt) =
        snapshot::read_snapshot_3d(&path).expect("lecture");
    assert_eq!((rnx, rny, rnz), (nx, ny, nz));
    assert_eq!(rstep, 1234);
    assert_eq!(rus, 0.02 * (ny - 1) as f64);
    // Identité bit-à-bit (f64 LE, aucune perte).
    assert_eq!(rrho, rho);
    assert_eq!(rux, ux);
    assert_eq!(ruy, uy);
    assert_eq!(ruz, uz);
    assert_eq!(rt, t);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn snapshot_3d_rejects_2d_magic() {
    // Un snapshot 2D (b"CAVSNAP1") doit être rejeté (format incompatible).
    let dir = std::env::temp_dir().join(format!("cav_snap3d_rej_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snap2d.bin");
    let mut data = b"CAVSNAP1".to_vec();
    data.extend_from_slice(&8u64.to_le_bytes());
    data.extend_from_slice(&8u64.to_le_bytes());
    std::fs::write(&path, &data).unwrap();
    let res = snapshot::read_snapshot_3d(&path);
    assert!(res.is_err(), "magic 2D doit être rejeté en 3D");
    std::fs::remove_dir_all(&dir).ok();
}

// ── Interpolation trilinéaire ─────────────────────────────────────────────

/// Champ linéaire f(x, y, z) = a + b·x + c·y + d·z échantillonné sur une
/// grille couvrant [0,1]³ (layout k = ix + nx·(iy + ny·iz)).
fn linear_field(nx: usize, ny: usize, nz: usize, a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(nx * ny * nz);
    for iz in 0..nz {
        let zf = if nz > 1 { iz as f64 / (nz - 1) as f64 } else { 0.5 };
        for iy in 0..ny {
            let yf = if ny > 1 { iy as f64 / (ny - 1) as f64 } else { 0.5 };
            for ix in 0..nx {
                let xf = if nx > 1 { ix as f64 / (nx - 1) as f64 } else { 0.5 };
                v.push(a + b * xf + c * yf + d * zf);
            }
        }
    }
    v
}

#[test]
fn interpolate_3d_same_grid_is_identity() {
    let (nx, ny, nz) = (12usize, 20usize, 9usize);
    let f = linear_field(nx, ny, nz, 0.3, 0.7, -0.2, 1.1);
    let out = snapshot::interpolate_field_3d(&f, nx, ny, nz, nx, ny, nz);
    // Identité exacte (early return src.to_vec()).
    assert_eq!(out, f);
}

#[test]
fn interpolate_3d_grid_change_preserves_linear_field_and_mean() {
    // Une fonction linéaire est reproduite exactement par l'interpolation
    // trilinéaire, et sa moyenne sur une grille uniforme couvrant [0,1] est
    // indépendante de la résolution (= valeur au centre) : la moyenne est
    // préservée au changement de grille.
    let (a, b, c, d) = (0.42, 1.3, -0.8, 0.55);
    let src = linear_field(16, 40, 10, a, b, c, d);
    let mean_src: f64 = src.iter().sum::<f64>() / src.len() as f64;

    // Upsampling + downsampling : même moyenne, valeurs interpolées exactes.
    for (dnx, dny, dnz) in [(24usize, 64usize, 16usize), (10usize, 20usize, 6usize)] {
        let dst = snapshot::interpolate_field_3d(&src, 16, 40, 10, dnx, dny, dnz);
        assert_eq!(dst.len(), dnx * dny * dnz);
        let mean_dst: f64 = dst.iter().sum::<f64>() / dst.len() as f64;
        assert!(
            (mean_dst - mean_src).abs() < 1e-10,
            "moyenne préservée : src={mean_src}, dst({dnx}×{dny}×{dnz})={mean_dst}"
        );
        // Chaque point interpolé = valeur linéaire exacte en ses coordonnées.
        let k_mid = dnx / 2 + dnx * (dny / 2 + dny * (dnz / 2));
        let xf = (dnx / 2) as f64 / (dnx - 1) as f64;
        let yf = (dny / 2) as f64 / (dny - 1) as f64;
        let zf = (dnz / 2) as f64 / (dnz - 1) as f64;
        let expect = a + b * xf + c * yf + d * zf;
        assert!((dst[k_mid] - expect).abs() < 1e-10, "valeur linéaire exacte : {} vs {}", dst[k_mid], expect);
    }
}

// ── Re-échelonnage vitesse (chaînage Ra, utarget 0.01 → 0.02) ────────────

#[test]
fn velocity_rescale_factor_is_correct() {
    // u_scale = utarget·(ny−1) : Ra=1e9 (utarget 0.01) → Ra=1e10 (utarget 0.02),
    // ny identique ⇒ facteur 2. Facteur calculé depuis les métadonnées du
    // snapshot (u_scale header), même formule que run_uniform3d.
    let (snx, sny, snz) = (16usize, 64usize, 16usize);
    let n = snx * sny * snz;
    let mut ux: Vec<f64> = (0..n).map(|k| 0.001 * k as f64).collect();
    let mut uy: Vec<f64> = (0..n).map(|k| -0.03 * k as f64).collect();
    let mut uz: Vec<f64> = (0..n).map(|k| 0.01 * k as f64).collect();
    let (ux0, uy0, uz0) = (ux.clone(), uy.clone(), uz.clone());

    let u_scale_snap = 0.01 * (sny - 1) as f64;
    let u_scale_new = 0.02 * (sny - 1) as f64; // ny destination identique
    let factor = u_scale_new / u_scale_snap;
    assert!((factor - 2.0).abs() < 1e-12);

    snapshot::rescale_velocity_fields(&mut ux, &mut uy, &mut uz, factor);
    for k in 0..n {
        assert!((ux[k] - 2.0 * ux0[k]).abs() < 1e-15);
        assert!((uy[k] - 2.0 * uy0[k]).abs() < 1e-15);
        assert!((uz[k] - 2.0 * uz0[k]).abs() < 1e-15);
    }

    // Cas utarget identique : facteur 1 (aucun changement).
    let factor1 = u_scale_snap / u_scale_snap;
    assert!((factor1 - 1.0).abs() < 1e-15);
}
