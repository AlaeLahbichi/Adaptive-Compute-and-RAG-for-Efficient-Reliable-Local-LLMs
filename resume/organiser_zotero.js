// =====================================================================
// Rangement de la collection Zotero du projet "Allocation adaptative..."
// À exécuter dans Zotero 7 : Outils → Développeur → Exécuter du JavaScript
// Cocher "Exécuter en tant que fonction asynchrone", coller ce script, puis "Exécuter".
//
// Ce que fait le script (il peut être relancé sans risque) :
//   1. crée 7 sous-collections thématiques dans la collection du projet
//   2. range chaque article dans son thème et ajoute un tag de priorité (P1…P4)
//   3. fusionne les 3 doublons (FrugalGPT, Hybrid LLM, Moskvoretskii ACL 2025)
//   4. supprime les 2 collections vides en double et l'ancienne sous-collection "Adaptive-RAG"
//   5. ajoute l'article manquant "Power Hungry Processing" (arXiv 2311.16863)
//   6. cherche les PDF manquants
// Rien n'est supprimé définitivement : tout ce qui est retiré va dans la corbeille Zotero.
// =====================================================================

const lib = Zotero.Libraries.userLibraryID;
const log = [];

// Collection principale du projet (celle qui contient les articles)
const PROJECT_KEY = "JBVZCMMB";
const EMPTY_DUPLICATE_COLLECTIONS = ["JWGGY5PB", "TVJ5Q3A7"];
const OLD_ADAPTIVE_RAG_SUBCOLLECTION = "MIGRFUCV";

// thème → [ [clé Zotero, priorité], ... ]
const THEMES = {
  "1. Routage & cascades de LLM": [
    ["A5K8VJXA", "P1"], ["R8ESBIKZ", "P1"], ["6JJMBCSC", "P1"], ["IJD56QW7", "P1"], ["PS2MRXPJ", "P1"],
    ["WUUCWLNC", "P2"], ["FCR8H4P4", "P2"], ["DDMM8VL9", "P2"],
    ["S73PMDFN", "P3"], ["MSU3SETV", "P3"], ["AVE2HYFG", "P3"],
  ],
  "2. RAG adaptatif": [
    ["QLWVKB9P", "P1"], ["9ZBRESJ3", "P1"], ["TFBGS7B7", "P1"], ["PGTJZ8X8", "P1"], ["YBTH2JX4", "P1"],
    ["C6VMM4JD", "P2"], ["9SGFDBCB", "P2"], ["YQA29UZD", "P2"], ["GXE3PHG4", "P2"],
    ["CFIMF2HZ", "P3"], ["PXR7FJP6", "P3"], ["HCZJL3SN", "P3"],
  ],
  "3. Réduction du contexte RAG": [
    ["7U7TJM8D", "P2"], ["F7TM8YI8", "P3"],
  ],
  "4. Incertitude & abstention": [
    ["YCMYEV8Q", "P1"], ["6UCVZ7UW", "P1"],
    ["NURBLKWC", "P2"], ["ENLLPZCI", "P2"], ["TPU4AQGY", "P2"],
    ["XMUL399M", "P3"],
  ],
  "5. Efficacité & énergie": [
    ["G8K3ALWB", "P1"], ["7LS37AUY", "P2"],
  ],
  "6. Benchmarks & données": [
    ["CHG8VTVT", "P1"], ["R2Q2B3F4", "P1"], ["7HXPLQDK", "P2"], ["QCYMJIXY", "P2"],
  ],
  "7. Périphérique": [
    ["MEGPAVAB", "P4"], ["4SBSPZ6R", "P4"], ["N7LE4N38", "P4"], ["ZY8H5ZB5", "P4"], ["SSANALRZ", "P4"],
  ],
};

// [clé de l'exemplaire conservé, [clés des doublons]]
const DUPLICATES = [
  ["R8ESBIKZ", ["FJHHFK4L"]], // FrugalGPT
  ["6JJMBCSC", ["M4PBR6JU"]], // Hybrid LLM (on garde la version ICLR)
  ["QLWVKB9P", ["D8L5LV5M"]], // Moskvoretskii, ACL 2025
];

const PRIORITY_TAGS = ["P1", "P2", "P3", "P4"];

const project = Zotero.Collections.getByLibraryAndKey(lib, PROJECT_KEY);
if (!project) throw new Error("Collection du projet introuvable (clé " + PROJECT_KEY + ")");

// ---------- 1. Fusion des doublons ----------
for (const [masterKey, otherKeys] of DUPLICATES) {
  const master = Zotero.Items.getByLibraryAndKey(lib, masterKey);
  const others = otherKeys.map(k => Zotero.Items.getByLibraryAndKey(lib, k)).filter(i => i && !i.deleted);
  if (!master || !others.length) continue;
  // récupérer le résumé du doublon si l'exemplaire conservé n'en a pas
  if (!master.getField("abstractNote")) {
    const withAbstract = others.find(o => o.getField("abstractNote"));
    if (withAbstract) {
      master.setField("abstractNote", withAbstract.getField("abstractNote"));
      await master.saveTx();
    }
  }
  try {
    await Zotero.Items.merge(master, others);
    log.push("Doublon fusionné : " + master.getField("title"));
  } catch (e) {
    log.push("⚠️ Fusion impossible pour " + master.getField("title") + " → à faire via 'Documents en double' (" + e + ")");
  }
}

// ---------- 2. Sous-collections thématiques ----------
async function getOrCreateSubcollection(name) {
  const existing = project.getChildCollections().find(c => c.name === name);
  if (existing) return existing;
  const c = new Zotero.Collection();
  c.libraryID = lib;
  c.name = name;
  c.parentID = project.id;
  await c.saveTx();
  log.push("Sous-collection créée : " + name);
  return c;
}

const themeCollections = {};
for (const theme of Object.keys(THEMES)) {
  themeCollections[theme] = await getOrCreateSubcollection(theme);
}

// ---------- 3. Rangement + tags de priorité ----------
let moved = 0;
for (const [theme, entries] of Object.entries(THEMES)) {
  const coll = themeCollections[theme];
  for (const [key, prio] of entries) {
    const item = Zotero.Items.getByLibraryAndKey(lib, key);
    if (!item || item.deleted) { log.push("⚠️ Introuvable : " + key); continue; }
    item.addToCollection(coll.id);
    item.removeFromCollection(project.id); // l'article vit maintenant dans son thème
    for (const t of PRIORITY_TAGS) if (t !== prio && item.hasTag(t)) item.removeTag(t);
    item.addTag(prio);
    if (prio !== "P4" && !item.hasTag("à lire") && !item.hasTag("lu")) item.addTag("à lire");
    await item.saveTx();
    moved++;
  }
}
log.push(moved + " articles rangés et tagués");

// ---------- 4. Nettoyage des collections en double ----------
for (const key of [...EMPTY_DUPLICATE_COLLECTIONS, OLD_ADAPTIVE_RAG_SUBCOLLECTION]) {
  const c = Zotero.Collections.getByLibraryAndKey(lib, key);
  if (!c || c.deleted) continue;
  const items = c.getChildItems(false);
  const onlyAdaptiveRag = key === OLD_ADAPTIVE_RAG_SUBCOLLECTION
    && items.every(i => i.key === "9ZBRESJ3");
  if ((items.length === 0 || onlyAdaptiveRag) && !c.hasChildCollections()) {
    await c.eraseTx(); // les articles ne sont pas supprimés, seulement la collection
    log.push("Collection supprimée (corbeille) : " + c.name + " [" + key + "]");
  } else {
    log.push("⚠️ Collection " + key + " non vide, conservée");
  }
}

// ---------- 5. Article manquant : Power Hungry Processing ----------
{
  const s = new Zotero.Search();
  s.libraryID = lib;
  s.addCondition("title", "contains", "Power Hungry Processing");
  s.addCondition("deleted", "false");
  const found = await s.search();
  if (found.length) {
    const it = Zotero.Items.get(found[0]);
    it.addToCollection(themeCollections["5. Efficacité & énergie"].id);
    it.addTag("P2"); it.addTag("à lire");
    await it.saveTx();
    log.push("Power Hungry Processing déjà présent → rangé");
  } else {
    try {
      const translate = new Zotero.Translate.Search();
      translate.setIdentifier({ arXiv: "2311.16863" });
      translate.setTranslator(await translate.getTranslators());
      const newItems = await translate.translate({
        libraryID: lib,
        collections: [themeCollections["5. Efficacité & énergie"].id],
        saveAttachments: true,
      });
      for (const it of newItems) { it.addTag("P2"); it.addTag("à lire"); await it.saveTx(); }
      log.push("Ajouté : Power Hungry Processing (arXiv 2311.16863)");
    } catch (e) {
      log.push("⚠️ Ajout de 2311.16863 impossible, à faire via la baguette magique (" + e + ")");
    }
  }
}

// ---------- 6. Recherche des PDF manquants ----------
let pdfFound = 0, pdfMissing = [];
for (const coll of Object.values(themeCollections)) {
  for (const item of coll.getChildItems(false)) {
    if (!item.isRegularItem() || item.itemType === "webpage") continue;
    const hasPdf = item.getAttachments()
      .map(id => Zotero.Items.get(id))
      .some(a => a && !a.deleted && a.attachmentContentType === "application/pdf");
    if (hasPdf) continue;
    try {
      const att = await Zotero.Attachments.addAvailableFile(item);
      if (att) pdfFound++; else pdfMissing.push(item.getField("title"));
    } catch (e) {
      pdfMissing.push(item.getField("title"));
    }
  }
}
log.push(pdfFound + " PDF téléchargés");
if (pdfMissing.length) log.push("PDF introuvables (à ajouter à la main) :\n  - " + pdfMissing.join("\n  - "));

return log.join("\n");
