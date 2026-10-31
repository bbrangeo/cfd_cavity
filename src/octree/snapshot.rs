//! Export/import de snapshots binaires MULTI-NIVEAUX pour l'octree
//! (Étape G — port du `snapshot.rs` du crate uniforme, adapté au layout
//! coarse L0 + blocs L1 + sous-niveaux L2 imbriqués).
//!
//! Format binaire (little-endian, f64 — même choix que le crate uniforme) :
//! ```text
//! [0..8)    magic  b"OCTSNAP1"
//! [8..12)   version u32 (= 1)
//! [12..16)  levels u32 (2 = config V2, 3 = escalier tanh V6)
//! [16..24)  step u64 (pas du solveur au moment de l'export)
//! [24..28)  n_grids u32 (nombre de grilles : 3 si levels=2, 5 si levels=3)
//! puis, pour CHAQUE grille (ordre : coarse L0, bloc chaud L1, bloc froid L1,
//! sous-niveau chaud L2, sous-niveau froid L2) :
//!   nx u64, ny u64
//! puis, pour CHAQUE grille dans le MÊME ordre, les champs consécutifs :
//!   rho (n·f64), ux (n·f64), uy (n·f64), temp (n·f64)
//! ```
//! L'export est ATOMIQUE (écriture temporaire + `std::fs::rename`, même
//! pattern que le crate uniforme) : si le processus meurt pendant l'écriture,
//! le fichier destination reste dans son état précédent.
//!
//! ⚠️ Étape G : reprise au MÊME layout uniquement — les dims de chaque
//! grille du snapshot doivent matcher exactement celles du solveur
//! (pas d'interpolation inter-grille, documenté dans la tâche).

use std::fs;
use std::io::Write;
use std::path::Path;

/// Magic binaire du snapshot octree (distinct du `CAVSNAP1` uniforme).
pub const MAGIC: &[u8; 8] = b"OCTSNAP1";

/// Version du format binaire.
pub const VERSION: u32 = 1;

/// Champs macroscopiques d'une grille du solveur (un niveau ou un bloc).
pub struct LevelFields {
    pub nx: usize,
    pub ny: usize,
    pub rho: Vec<f64>,
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    pub temp: Vec<f64>,
}

/// Snapshot multi-niveaux complet : niveaux + pas associé.
pub struct Snapshot {
    /// Nombre de niveaux du solveur source (2 ou 3).
    pub levels: usize,
    /// Pas du solveur au moment de l'export.
    pub step: u64,
    /// Grilles dans l'ordre canonique : coarse, bloc chaud, bloc froid,
    /// [sous-niveau chaud, sous-niveau froid si levels=3].
    pub grids: Vec<LevelFields>,
}

/// Sérialise le snapshot en bytes (little-endian, f64).
pub fn encode(snap: &Snapshot) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(snap.levels as u32).to_le_bytes());
    buf.extend_from_slice(&snap.step.to_le_bytes());
    buf.extend_from_slice(&(snap.grids.len() as u32).to_le_bytes());
    // Dims de chaque grille.
    for g in &snap.grids {
        buf.extend_from_slice(&(g.nx as u64).to_le_bytes());
        buf.extend_from_slice(&(g.ny as u64).to_le_bytes());
    }
    // Champs : rho, ux, uy, temp par grille, dans le même ordre.
    for g in &snap.grids {
        for v in g.rho.iter().chain(&g.ux).chain(&g.uy).chain(&g.temp) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Désérialise un snapshot depuis des bytes. Longueur STRICTEMENT vérifiée
/// (dims déclarées vs taille des données) — erreur claire sinon.
pub fn decode(data: &[u8]) -> std::io::Result<Snapshot> {
    let err = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);
    if data.len() < 28 || &data[..8] != MAGIC {
        return Err(err("snapshot octree : magic invalide ou fichier trop court".into()));
    }
    let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(err(format!(
            "snapshot octree : version {version} non supportée (attendu {VERSION})"
        )));
    }
    let levels = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let step = u64::from_le_bytes(data[16..24].try_into().unwrap());
    let n_grids = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
    if !(levels == 2 || levels == 3) || n_grids != if levels == 2 { 3 } else { 5 } {
        return Err(err(format!(
            "snapshot octree : incohérence levels={levels} / n_grids={n_grids} (attendu 3/5)"
        )));
    }
    // Dims.
    let dims_end = 28 + n_grids * 16;
    if data.len() < dims_end {
        return Err(err("snapshot octree : header dims tronqué".into()));
    }
    let mut dims = Vec::with_capacity(n_grids);
    let mut expect = dims_end;
    for gi in 0..n_grids {
        let off = 28 + gi * 16;
        let nx = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        let ny = u64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap()) as usize;
        expect += 4 * nx * ny * 8;
        dims.push((nx, ny));
    }
    if data.len() != expect {
        return Err(err(format!(
            "snapshot octree : attendu {expect} octets, lu {}",
            data.len()
        )));
    }
    // Champs.
    let mut grids = Vec::with_capacity(n_grids);
    let mut off = dims_end;
    for (nx, ny) in dims {
        let n = nx * ny;
        let read_field = |off: &mut usize| -> Vec<f64> {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(f64::from_le_bytes(
                    data[*off + i * 8..*off + (i + 1) * 8].try_into().unwrap(),
                ));
            }
            *off += n * 8;
            v
        };
        let rho = read_field(&mut off);
        let ux = read_field(&mut off);
        let uy = read_field(&mut off);
        let temp = read_field(&mut off);
        grids.push(LevelFields { nx, ny, rho, ux, uy, temp });
    }
    Ok(Snapshot { levels, step, grids })
}

/// Écrit un snapshot binaire de manière ATOMIQUE : écriture dans un fichier
/// temporaire du même répertoire, puis rename vers `path` (même pattern que
/// le crate uniforme — jamais de fichier destination corrompu).
pub fn write_snapshot_atomic(path: &Path, snap: &Snapshot) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().unwrap_or_default();
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = dir.join(tmp_name);
    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(&encode(snap))?;
        f.sync_all()?;
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        // Nettoyage du temporaire si le rename échoue (ex. cross-device).
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

/// Lit un snapshot binaire depuis un fichier.
pub fn read_snapshot(path: &Path) -> std::io::Result<Snapshot> {
    let data = fs::read(path)?;
    decode(&data)
}
