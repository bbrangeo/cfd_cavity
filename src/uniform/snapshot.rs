//! Export/import de snapshots binaires pour le chaînage Ra élevé
//! (ex. Ra=1e8 équilibré → Ra=1e10), avec interpolation bilinéaire
//! si la grille de destination diffère.

use std::fs;
use std::io::Write;
use std::path::Path;

/// En-tête binaire du snapshot (magic + versions + grille).
const MAGIC: &[u8; 8] = b"CAVSNAP1";

/// Écrit un snapshot binaire : rho, ux, uy, T (ordre nœud (ix, iy),
/// k = iy*nx + ix, même convention que le solveur).
pub fn write_snapshot(
    path: &Path,
    nx: usize,
    ny: usize,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    t: &[f64],
) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&(nx as u64).to_le_bytes())?;
    f.write_all(&(ny as u64).to_le_bytes())?;
    for v in rho.iter().chain(ux).chain(uy).chain(t) {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Écrit un snapshot binaire de manière ATOMIQUE : les données sont d'abord
/// écrites dans un fichier temporaire du même répertoire, puis renommées vers
/// `path` (std::fs::rename). Si le processus meurt pendant l'écriture, le
/// fichier destination reste dans son état précédent (jamais corrompu).
pub fn write_snapshot_atomic(
    path: &Path,
    nx: usize,
    ny: usize,
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    t: &[f64],
) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().unwrap_or_default();
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = dir.join(tmp_name);
    write_snapshot(&tmp_path, nx, ny, rho, ux, uy, t)?;
    if let Err(e) = fs::rename(&tmp_path, path) {
        // Nettoyage du temporaire si le rename échoue (ex. cross-device).
        let _ = fs::remove_file(tmp_path);
        return Err(e);
    }
    Ok(())
}

/// Lit un snapshot binaire. Retourne (nx, ny, rho, ux, uy, t).
pub fn read_snapshot(path: &Path) -> std::io::Result<(usize, usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let data = fs::read(path)?;
    if data.len() < 24 || &data[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot : magic invalide ou fichier trop court",
        ));
    }
    let nx = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let ny = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
    let n = nx * ny;
    let expect = 24 + 4 * n * 8;
    if data.len() < expect {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("snapshot : attendu {expect} octets, lu {}", data.len()),
        ));
    }
    let mut rho = Vec::with_capacity(n);
    let mut ux = Vec::with_capacity(n);
    let mut uy = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut off = 24;
    let mut read_field = |v: &mut Vec<f64>| {
        for i in 0..n {
            v.push(f64::from_le_bytes(data[off + i * 8..off + (i + 1) * 8].try_into().unwrap()));
        }
        off += n * 8;
    };
    read_field(&mut rho);
    read_field(&mut ux);
    read_field(&mut uy);
    read_field(&mut t);
    Ok((nx, ny, rho, ux, uy, t))
}

/// Interpole bilinéairement un champ d'une grille (nx0, ny0) vers (nx1, ny1).
/// Les coordonnées physiques sont normalisées : x ∈ [0,1], y ∈ [0,1]
/// (les deux domaines représentent la même cavité Az=4).
pub fn interpolate_field(src: &[f64], nx0: usize, ny0: usize, nx1: usize, ny1: usize) -> Vec<f64> {
    if (nx0, ny0) == (nx1, ny1) {
        return src.to_vec();
    }
    let mut dst = Vec::with_capacity(nx1 * ny1);
    for iy in 0..ny1 {
        // Coordonnée normalisée y de la destination.
        let vy = if ny1 > 1 { iy as f64 / (ny1 - 1) as f64 } else { 0.0 };
        // Position correspondante dans la source (indices fractionnaires).
        let sy = vy * (ny0 - 1) as f64;
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(ny0 - 1);
        let wy = sy - y0 as f64;
        for ix in 0..nx1 {
            let vx = if nx1 > 1 { ix as f64 / (nx1 - 1) as f64 } else { 0.0 };
            let sx = vx * (nx0 - 1) as f64;
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(nx0 - 1);
            let wx = sx - x0 as f64;
            let v00 = src[y0 * nx0 + x0];
            let v10 = src[y0 * nx0 + x1];
            let v01 = src[y1 * nx0 + x0];
            let v11 = src[y1 * nx0 + x1];
            let val = v00 * (1.0 - wx) * (1.0 - wy)
                + v10 * wx * (1.0 - wy)
                + v01 * (1.0 - wx) * wy
                + v11 * wx * wy;
            dst.push(val);
        }
    }
    dst
}
