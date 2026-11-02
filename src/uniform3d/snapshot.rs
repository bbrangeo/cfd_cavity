//! Export/import de snapshots binaires 3D pour le chaînage Ra élevé
//! (ex. Ra=1e9 équilibré → Ra=1e10), avec interpolation trilinéaire
//! si la grille de destination diffère.
//!
//! ⚠️ Format INDÉPENDANT du snapshot 2D (`uniform/snapshot.rs`, magic
//! b"CAVSNAP1", nx/ny + 4 champs) : magic b"SNP3", header étendu
//! (nx/ny/nz/step en u64 LE + u_scale en f64 LE) et 5 champs f64 LE
//! (rho, ux, uy, uz, T). Aucun code partagé avec le 2D — un snapshot
//! 2D rechargé en 3D (ou l'inverse) échoue sur le magic.
//!
//! `u_scale` (u_max cible en unités réseau du run source, = utarget·(ny−1))
//! permet le re-échelonnage des vitesses au restart : quand on redémarre un
//! snapshot Ra=1e9 (utarget 0.01) vers Ra=1e10 (utarget 0.02), les unités de
//! réseau changent et le champ u doit être multiplié par
//! u_scale_new / u_scale_snap. T est adimensionnelle et ρ ≈ 1 : inchangés.

use std::fs;
use std::io::Write;
use std::path::Path;

/// En-tête binaire du snapshot 3D (magic 4 octets).
const MAGIC: &[u8; 4] = b"SNP3";

/// Taille du header : magic (4) + nx/ny/nz/step (4 × u64) + u_scale (f64).
const HEADER_LEN: usize = 4 + 4 * 8 + 8;

/// Écrit un snapshot binaire 3D : rho, ux, uy, uz, T (ordre nœud
/// k = ix + nx·(iy + ny·iz), même convention que `CavitySolver3D`).
///
/// `u_scale` = u_max cible du run en unités de réseau (utarget·(ny−1)) :
/// métadonnée de re-échelonnage lue par `read_snapshot_3d`.
#[allow(clippy::too_many_arguments)]
pub fn write_snapshot_3d(
    path: &Path,
    nx: usize,
    ny: usize,
    nz: usize,
    step: u64,
    u_scale: f64,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    t: &[f64],
) -> std::io::Result<()> {
    let n = nx * ny * nz;
    debug_assert_eq!(rho.len(), n, "rho : taille attendue n = nx·ny·nz");
    debug_assert_eq!(ux.len(), n, "ux : taille attendue n = nx·ny·nz");
    debug_assert_eq!(uy.len(), n, "uy : taille attendue n = nx·ny·nz");
    debug_assert_eq!(uz.len(), n, "uz : taille attendue n = nx·ny·nz");
    debug_assert_eq!(t.len(), n, "t : taille attendue n = nx·ny·nz");
    let mut f = fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&(nx as u64).to_le_bytes())?;
    f.write_all(&(ny as u64).to_le_bytes())?;
    f.write_all(&(nz as u64).to_le_bytes())?;
    f.write_all(&step.to_le_bytes())?;
    f.write_all(&u_scale.to_le_bytes())?;
    for v in rho.iter().chain(ux).chain(uy).chain(uz).chain(t) {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Écrit un snapshot binaire 3D de manière ATOMIQUE : les données sont
/// d'abord écrites dans un fichier temporaire du même répertoire, puis
/// renommées vers `path` (std::fs::rename). Si le processus meurt pendant
/// l'écriture, le fichier destination reste dans son état précédent
/// (jamais corrompu) — même motif que le 2D `write_snapshot_atomic`.
#[allow(clippy::too_many_arguments)]
pub fn write_snapshot_3d_atomic(
    path: &Path,
    nx: usize,
    ny: usize,
    nz: usize,
    step: u64,
    u_scale: f64,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    uz: &[f64],
    t: &[f64],
) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().unwrap_or_default();
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = dir.join(tmp_name);
    write_snapshot_3d(&tmp_path, nx, ny, nz, step, u_scale, rho, ux, uy, uz, t)?;
    if let Err(e) = fs::rename(&tmp_path, path) {
        // Nettoyage du temporaire si le rename échoue (ex. cross-device).
        let _ = fs::remove_file(tmp_path);
        return Err(e);
    }
    Ok(())
}

/// Lit un snapshot binaire 3D. Retourne
/// (nx, ny, nz, step, u_scale, rho, ux, uy, uz, t).
#[allow(clippy::type_complexity)]
pub fn read_snapshot_3d(
    path: &Path,
) -> std::io::Result<(
    usize,
    usize,
    usize,
    u64,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let data = fs::read(path)?;
    if data.len() < HEADER_LEN || &data[..4] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot 3D : magic invalide (b\"SNP3\" attendu) ou fichier trop court \
             (un snapshot 2D b\"CAVSNAP1\" n'est pas compatible)",
        ));
    }
    let nx = u64::from_le_bytes(data[4..12].try_into().unwrap()) as usize;
    let ny = u64::from_le_bytes(data[12..20].try_into().unwrap()) as usize;
    let nz = u64::from_le_bytes(data[20..28].try_into().unwrap()) as usize;
    let step = u64::from_le_bytes(data[28..36].try_into().unwrap());
    let u_scale = f64::from_le_bytes(data[36..44].try_into().unwrap());
    let n = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "snapshot 3D : grille nx·ny·nz déborde (header corrompu ?)",
            )
        })?;
    let expect = HEADER_LEN + 5 * n * 8;
    if data.len() < expect {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("snapshot 3D : attendu {expect} octets, lu {}", data.len()),
        ));
    }
    let mut rho = Vec::with_capacity(n);
    let mut ux = Vec::with_capacity(n);
    let mut uy = Vec::with_capacity(n);
    let mut uz = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut off = HEADER_LEN;
    let mut read_field = |v: &mut Vec<f64>| {
        for i in 0..n {
            v.push(f64::from_le_bytes(
                data[off + i * 8..off + (i + 1) * 8].try_into().unwrap(),
            ));
        }
        off += n * 8;
    };
    read_field(&mut rho);
    read_field(&mut ux);
    read_field(&mut uy);
    read_field(&mut uz);
    read_field(&mut t);
    Ok((nx, ny, nz, step, u_scale, rho, ux, uy, uz, t))
}

/// Interpole trilinéairement un champ 3D d'une grille (snx, sny, snz) vers
/// (dnx, dny, dnz). Les coordonnées physiques sont normalisées : x, y, z ∈
/// [0,1] (les deux domaines représentent la même cavité AR ≈ 4).
///
/// Logique adaptée du 2D `interpolate_field` (bilinear) : en x et y les
/// coordonnées couvrent les parois (isothermes/adiabatiques), en z la
/// normalisation [0,1] est celle d'une grille périodique sans wrap (le
/// domaine source et destination partagent l'origine iz = 0).
///
/// Layout source ET destination : k = ix + nx·(iy + ny·iz) (convention
/// `CavitySolver3D`, DIFFÉRENTE du 2D qui utilise iy*nx + ix).
pub fn interpolate_field_3d(
    src: &[f64],
    snx: usize,
    sny: usize,
    snz: usize,
    dnx: usize,
    dny: usize,
    dnz: usize,
) -> Vec<f64> {
    if (snx, sny, snz) == (dnx, dny, dnz) {
        return src.to_vec();
    }
    let mut dst = Vec::with_capacity(dnx * dny * dnz);
    for iz in 0..dnz {
        // Coordonnée normalisée z de la destination.
        let vz = if dnz > 1 { iz as f64 / (dnz - 1) as f64 } else { 0.0 };
        // Position correspondante dans la source (indices fractionnaires).
        let sz = vz * (snz - 1) as f64;
        let z0 = sz.floor() as usize;
        let z1 = (z0 + 1).min(snz - 1);
        let wz = sz - z0 as f64;
        for iy in 0..dny {
            let vy = if dny > 1 { iy as f64 / (dny - 1) as f64 } else { 0.0 };
            let sy = vy * (sny - 1) as f64;
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(sny - 1);
            let wy = sy - y0 as f64;
            for ix in 0..dnx {
                let vx = if dnx > 1 { ix as f64 / (dnx - 1) as f64 } else { 0.0 };
                let sx = vx * (snx - 1) as f64;
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(snx - 1);
                let wx = sx - x0 as f64;
                // 8 coins (trilinéaire), layout k = ix + nx·(iy + ny·iz).
                let c = |szl: usize, syl: usize, sxl: usize| -> f64 {
                    src[szl * snx * sny + syl * snx + sxl]
                };
                let val = c(z0, y0, x0) * (1.0 - wx) * (1.0 - wy) * (1.0 - wz)
                    + c(z0, y0, x1) * wx * (1.0 - wy) * (1.0 - wz)
                    + c(z0, y1, x0) * (1.0 - wx) * wy * (1.0 - wz)
                    + c(z0, y1, x1) * wx * wy * (1.0 - wz)
                    + c(z1, y0, x0) * (1.0 - wx) * (1.0 - wy) * wz
                    + c(z1, y0, x1) * wx * (1.0 - wy) * wz
                    + c(z1, y1, x0) * (1.0 - wx) * wy * wz
                    + c(z1, y1, x1) * wx * wy * wz;
                dst.push(val);
            }
        }
    }
    dst
}

/// Re-échelonne les champs de vitesse (u_new = u_old × factor) au restart.
/// Factor = u_scale_new / u_scale_snap (u_scale = utarget·(ny−1)) : les
/// unités de réseau changent quand utarget change (chaînage Ra, le ν reste
/// calé pareil mais la vitesse cible diffère). T est adimensionnelle et
/// ρ ≈ 1 : non touchés.
pub fn rescale_velocity_fields(ux: &mut [f64], uy: &mut [f64], uz: &mut [f64], factor: f64) {
    for v in ux.iter_mut().chain(uy.iter_mut()).chain(uz.iter_mut()) {
        *v *= factor;
    }
}
