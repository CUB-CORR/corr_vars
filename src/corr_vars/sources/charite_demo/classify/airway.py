"""Classify charite_demo airway-device records into canonical device categories.

Shipped as package code and declared in
:data:`corr_vars.sources.charite_demo.py_env.IMPORT_NAMES`, so a served snippet finds
``classify_airway_devices`` in its namespace.

TODO: split the data from the logic. The curated lookup tables below (``CLASSES``
and the pattern/synonym tables) are *data*, not logic, and belong in the
versioned file library alongside the other mapping files, reached with
``getfile("<uuid>")``. As long as they live here, correcting a single row of a
table needs a corr-vars release and no concept version records the change, so
the correction neither versions nor cascades the way every other mapping file's
does.
"""

from __future__ import annotations

import polars as pl

from typing import Final

__all__ = ["classify_airway_devices"]

# --------------------------------------------------------------------------
# Output classes (15 canonical labels used downstream)
# --------------------------------------------------------------------------
CLASSES: Final = {
    "ETT": "Endotracheal tube",
    "MCETT": "Microcuff Endotracheal tube",
    "TT": "Tracheal tube",
    "ETT_TS": "Endotracheal tube inserted through tracheostoma",
    "LT": "Laryngektomietubus",
    "LMA": "Laryngeal mask airway",
    "NIV": "Non-Invasive Ventilation Mask (NIV)",
    "HFNC": "High Flow nasal cannula interface",
    "O2NC": "Oxygen nasal cannula",
    "O2M": "Oxygen mask",
    "OPA": "Oropharyngeal airway",
    "NPT": "Nasopharyngeal tube",
    "JET": "Jet ventilation catheter",
    "TSP": "Tracheostoma placeholder",
    "UNCLEAR": "Combination unclear. Categorization not possible.",
}

SENTINEL: Final = "__NULL__"


# --------------------------------------------------------------------------
# Product-family / stem patterns
# --------------------------------------------------------------------------
TK_STEMS_HARD: Final = (
    r"^tk\b|^tk | tk\b|-tk\b|^spiraltk|^spiral tk|^spiral-tk|"
    r"trachealka|tracheoflex|silikon-?tk|short tk|hirtenstab tk|"
    r"transtrachtk|piriflex tk|periflex|primed proline tk|primed"
)
TK_FAMILY_OTHER: Final = (
    r"tracoe|ultra tk|extra tk|tracflex|priflex|bivona|ultrasoft|sealguard|"
    r"diflex|triflex"
)

NIV_FAMILY: Final = (
    r"\bmaske\b|nasenmaske|oronasal|\bniv\b|cpap|bipap|fullface|full.?face|"
    r"halbgesichtsmaske|halbesichtsmaske|gesichtsmaske|gesichtmaske|atemmaske|"
    r"intubationsmaske|kinnband|"
    r"^ff|^tff|^e-ma|^n-ma|^e-ac|^e-af|^e-ff|^e-vis|^e-es|^e-ni|^e-per|^e-ev|^e-vit|"
    r"^n-af|^n-eson|^n-vis|^n-ni|^n-wisp|^n-sfx|^n-one|^n-nua|^n-cara|^n-m|^n-mu|"
    r"^mu-n|^m-n-|^mn-ma|^n/|^nasen-|^nasen/|"
    r"mirage|mirrage|amara|performax|perfomax|perfomask|performatrak|"
    r"e-nivairo|nivairo|nivario|e-perfomax|evora|quattro|quatro|dreamwear|"
    r"airfit|air ?fit|air ?ma|"
    r"visairo|visario|vitera|vitairo|evaro|evairo|evor\b|evor ffv|"
    r"wispv|sfxv|esson|eson|acucare|accu care|ecu care|acu care|"
    r"resmed|res med|rsmed|rs med|respironic|philips|fisher|fischer|f&p|fp visairo|"
    r"joyce|silkgel|silk gel|simplus|sim ?plus|thrive|stephan system|"
    r"pegaso|weimann|cara|nivmaske|nivbrille|nivario|individualmaske|presmed|"
    r"c-pap|full face|total face|joyce fullface"
)

ETT_STEMS: Final = (
    r"\btubus\b|spiraltu|doppellu|doppeltu|magill|vygon|vigon|viggon|vyggon|vyg\.|"
    r"yugon|ygon|\bmlt\b|"
    r"microcuff|micro-?cuff|mikro-?cuff|mikrocuff|micocuff|mikocuff|microcuf|"
    r"mikto cuff|mikro tubus|mikrotubus|mikrolaryngealtubus|mikro\b|gebl mikro|"
    r"tritube|tritubus|tri-tube|tri tube|tribute|"
    r"\bdlt\b|rae|elektrodentu|elektrodentubus|stimulationstubus|emg stim|"
    r"high.?press.?cuff|c-high-press|woodbridge|woodbrigde|c-woodbridge|"
    r"bronchoflex|bronchioflex|bronchoskopietubus|narkosetubus|"
    r"laryngoflex|kleinsasser|laryngsmaske|larynxaske|"
    r"evone|endotrachealtubus|tubu with|nasaler|nasaltubus|nasentubus|"
    r"rueschelit|nasal sefety|rusch nasotrach|"
    r"portex polar|portex|mallinckrodt|mallinkrodt|mallinckodt|mallinchrodt|"
    r"malinnckrodt|malinckrodt|c-mallinckrodt|c-malinkrodt|c-portex|"
    r"shiley|shileytubus|"
    r"starres rohr|starres bronchoskop|bronchusblock gebl|medronic nro|frova"
)

LT_STEMS: Final = r"laryngektom"  # kept separate; Larynxtubus/LGT/LaTu go to LMA family

NPT_STEMS: Final = (
    r"rachentubus|rachentu|wendeltu|wendltu|wendltubus|c-wendel|wendel nasal|"
    r"wendel [0-9]|wendel \d+[a-z]*|"
    r"rachen tu|ratubus|rachntubus|^wendel|wendel rechts|wendel links|"
    r"nasopharyng|choanalroehr|nasen-?roehrchen|roehrchen|nasen/ ?roehrchen"
)

HFNC_STEMS: Final = (
    r"optiflow|opti-?flow|opti flow|hfnc|hi-?flow|high-? ?flow|hfo|"
    r"nasales interface|nasal.?interface|"
    r"\bhf\b|hf-?system|hfb|ram-?cann|ram cannula|thrive|"
    r"higflow|higflowbrille|feuchte nase|freuchte nase|hf - neonaten"
)

O2M_STEMS: Final = (
    r"venturi|o2.?maske|sauerstoffmaske|reservoir|\bnrb\b|"
    r"o2.?beutelmaske|o2beutelmaske|o2-?beutelmaske|beutelmaske|"
    r"o2 inufflation|ambu.?beutel"
)

O2NC_STEMS: Final = (
    r"freddy|inhalett|inhalete|inhallet|inhalat\.?|inhaletette|"
    r"inhalationsystem|inhalationssystem|inhalationsset|inhalation system|"
    r"o2-?\s?brille|hf-?brille|sauerstoffbrille|saerstoffbrille|sauersoffbrille|"
    r"nasenbrille|nasalbrille|low-?flow-?brille|o2brill|o2briile|o2-brill|"
    r"02brille|o2-?erwachsenenbrille|o2-?sonde|o2-?nasensonde|o2-?katheter|"
    r"nasensonde|n-brille|nivbrille|"
    r"low-?flow|lowflow|low flow|"
    r"\bbrille\b|o2 vorlage|o2-?vorlage|^vorlage$|masken vorlage|"
    r"ambubeutel vorlage|ambu beutel vorlage"
)

ETT_BRAND_STEMS: Final = (
    r"^portex$|^mallinckrodt$|^shiley$|^rusch$|^ruesch$|^rueschelit$"
)

MC_STEMS = (
    r"microcuff|micro-?cuff|mikro-?cuff|mikrocuff|micocuff|mikocuff|"
    r"microcuf|mikto cuff|mikro tubus|mikrotubus|mikrolaryngealtubus|"
    r"tritube|tritubus|tri-tube|tri tube|tribute|\bmlt\b|"
    r"rae.*microcuff|rae.*mikrocuff"
)


def normalize(expr: pl.Expr) -> pl.Expr:
    """Lowercase, trim, collapse whitespace, remove umlauts, handle ² -> 2."""
    return (
        expr.fill_null("")
        .cast(pl.Utf8)
        .replace(SENTINEL, "")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.replace_all("²", "2")
        .str.replace_all("ä", "ae")
        .str.replace_all("ö", "oe")
        .str.replace_all("ü", "ue")
        .str.replace_all("ß", "ss")
        .str.strip_chars()
    )


def classification_struct(
    cl: pl.Expr | str, conf: pl.Expr | str, rule: pl.Expr | str
) -> pl.Expr:
    """Helper to wrap classification results into a Polars struct."""
    return pl.struct(
        cl=cl if isinstance(cl, pl.Expr) else pl.lit(cl),
        conf=conf if isinstance(conf, pl.Expr) else pl.lit(conf),
        rule=rule if isinstance(rule, pl.Expr) else pl.lit(rule),
    )


def classify_airway_devices(
    df: pl.DataFrame,
    typ_col: str = "typ",
    app_col: str = "applika",
    anl_col: str = "anlageart",
) -> pl.DataFrame:
    """Airway device auto-classifier (v8).

    Takes a table with columns (typ_col, app_col, anl_col)
    and assigns each row to one of 15 airway-device classes, with a confidence level
    (very_high / high / medium / low / unclear) and the name of the rule that fired.

    Rules were derived from the Charité free-text variants and reviewed clinically.
    Rule PRIORITY matters — rules are tried in order; first match wins. Ordering:
    1. Placeholders (always TSP)
    2. Laryngectomy tube (explicit)
    3. TK-family (hard TK override — runs before LMA)
    4. Speaking cannula → TK
    5. Non-airway / airway-clearance / junk → unclear (so they don't poison later rules)
    6. LMA / Larynx tube family
    7. ETT stems (runs before bronchial-blocker so "Tubus + Bronchblock" → ETT)
    8. Pure bronchial blocker (without a tube word) → unclear
    9. Subglottic suction marker
    10. Tracheostoma-context default → TK
    11. Blueline / ETT brand names resolved by context
    12. Nasopharyngeal / Wendl / Rachen family
    13. Oropharyngeal (Guedel, oral prosthetics)
    14. Jet / transtracheal
    15. Prongs (NIV if explicit, else HFNC)
    16. HFNC product families
    17. O2 masks (low flow)
    18. O2 nasal cannula (low flow — "Brille" family)
    19. NIV mask family (product codes + brands)
    20. Context-only fallback when typ is empty

    Args:
        df (pl.DataFrame): Table with columns (typ_col, app_col, anl_col).
        typ_col (str): Column name for typ.
        app_col (str): Column name for applika.
        anl_col (str): Column name for anlageart.

    Returns:
        pl.DataFrame: Table with columns (auto_class, auto_conf, auto_rule).
    """
    # 1. Preprocessing & Context
    df = df.with_columns(
        __typ_n__=pl.col(typ_col).pipe(normalize),
        __app_n__=pl.col(app_col).pipe(normalize),
        __anl_n__=pl.col(anl_col).pipe(normalize),
    ).with_columns(
        ts_ctx=pl.any_horizontal(
            pl.col("__app_n__", "__anl_n__").str.contains(
                r"tracheostom|tracheotom|transtrach|coniotom|koniotom"
            )
        ),
        oral_ctx=pl.col("__app_n__").str.contains(r"orotracheal|nasotracheal|\boral\b")
        | pl.col("__anl_n__").str.contains("intub"),
    )

    TYP, APP, TS, ORAL = (
        pl.col("__typ_n__"),
        pl.col("__app_n__"),
        pl.col("ts_ctx"),
        pl.col("oral_ctx"),
    )

    # 2. Classification Logic
    expr = (
        # 1. Placeholders
        pl.when(
            TYP.str.contains(
                r"platzhalter|placeholder|stoma.?button|\bplatzh\b|^stomab\b"
            )
        )
        .then(
            classification_struct(
                CLASSES["TSP"], "very_high", "stoma_placeholder_any_ctx"
            )
        )
        # 2. Laryngectomy tube
        .when(TYP.str.contains("laryngektom"))
        .then(classification_struct(CLASSES["LT"], "very_high", "laryngectomy_tube"))
        # 3. TK family (hard override — runs BEFORE LMA so "TK Rüsch / supraglottisch" → TK)
        .when(TYP.str.contains(TK_STEMS_HARD))
        .then(classification_struct(CLASSES["TT"], "very_high", "TK_hard_stem"))
        .when(TYP.str.contains(TK_FAMILY_OTHER))
        .then(classification_struct(CLASSES["TT"], "very_high", "TK_family"))
        # 4. Speaking cannula on a trach
        .when(TYP.str.contains(r"sprechkanuele|sprecha|spreckanuele"))
        .then(classification_struct(CLASSES["TT"], "high", "speaking_cannula→TK"))
        # 5a. Non-airway catheters mis-charted in AWZ
        .when(
            TYP.str.contains(
                r"oesophagustemperatur|temperatursonde|temperaturkath|terumo|flexuele|cook exchange|cook extubation|cook stab|cook-stab|exchangekath|exchange kath|^exchange\b|fuehrungsdraht"
            )
        )
        .then(classification_struct(CLASSES["UNCLEAR"], "unclear", "non_airway_device"))
        # 5b. Airway-clearance (not an interface)
        .when(
            TYP.str.contains(
                r"cough.?assist|ezpap|ez pap|triflow|\bblubber\b|aero ?chamber|inhalationsset pari|pari boy"
            )
        )
        .then(
            classification_struct(
                CLASSES["UNCLEAR"], "unclear", "airway_clearance_not_interface"
            )
        )
        # 5c. Data-quality junk
        .when(
            TYP.str.contains(
                r"^(k\.?a\.?|ka|keine|fehleingabe|fehldokumentation|-|n\.z\.|o|binasal|blubber|ts|fp visairo|kinnband)$"
            )
        )
        .then(classification_struct(CLASSES["UNCLEAR"], "unclear", "data_quality_junk"))
        # 6. LMA (by typ evidence, or supraglottisch-applika when typ empty)
        .when(
            TYP.str.contains(r"\blma\b|^lama|^la ma|larynxmaske|laryngeal mask|i-?gel")
        )
        .then(classification_struct(CLASSES["LMA"], "very_high", "LMA_typ"))
        .when((APP.str.contains("supraglotti")) & (TYP == ""))
        .then(classification_struct(CLASSES["LMA"], "high", "LMA_supraglottic_app"))
        .when(TYP.str.contains(r"^larynxtubus|\blgt\b|^latu\b|laryngsmaske|larynxaske"))
        .then(classification_struct(CLASSES["LMA"], "high", "larynx_tube_family"))
        # 7. ETT stems (runs BEFORE bronchial-blocker rule so "Tubus + Bronchblock" → ETT)
        .when(TYP.str.contains(ETT_STEMS))
        .then(
            classification_struct(
                cl=pl.when(TS)
                .then(pl.lit(CLASSES["ETT_TS"]))
                .when(TYP.str.contains(MC_STEMS))
                .then(pl.lit(CLASSES["MCETT"]))
                .otherwise(pl.lit(CLASSES["ETT"])),
                conf="very_high",
                rule=pl.when(TS)
                .then(pl.lit("ETT_stem_via_tracheostoma"))
                .when(TYP.str.contains(MC_STEMS))
                .then(pl.lit("microcuff_ETT"))
                .otherwise(pl.lit("ETT_stem")),
            )
        )
        # 8. Pure bronchial blocker (no tube word present)
        .when(
            TYP.str.contains(
                r"bronchblock|bronchusblock|bonchusblocker|ez-?blocker|ez-?block|endobronchial.?block|bronchial block|cohen endobronch"
            )
        )
        .then(
            classification_struct(
                CLASSES["UNCLEAR"], "unclear", "bronchial_blocker_not_awz"
            )
        )
        # 9. Subglottic suction marker
        .when(TYP.str.contains(r"\bsgla\b|\bsub.?glott"))
        .then(
            classification_struct(
                cl=pl.when(ORAL)
                .then(pl.lit(CLASSES["ETT"]))
                .when(TS)
                .then(pl.lit(CLASSES["TT"]))
                .otherwise(pl.lit(CLASSES["UNCLEAR"])),
                conf="very_high",
                rule=pl.when(ORAL)
                .then(pl.lit("sglA_ETT"))
                .otherwise(pl.lit("sglA_in_TS_ctx")),
            )
        )
        # 10. Tracheostoma-context default → TK
        .when(TS)
        .then(classification_struct(CLASSES["TT"], "very_high", "TS_context→TK"))
        # 11. Brands/Blueline (Inner Branch)
        .when(
            TYP.str.contains(r"blueline|blue.?line|bluline")
            | TYP.str.contains(ETT_BRAND_STEMS)
        )
        .then(
            classification_struct(
                cl=pl.when(ORAL)
                .then(pl.lit(CLASSES["ETT"]))
                .otherwise(pl.lit(CLASSES["UNCLEAR"])),
                conf=pl.when(ORAL)
                .then(pl.lit("very_high"))
                .otherwise(pl.lit("medium")),
                rule=pl.when(TYP.str.contains(r"blueline|blue.?line|bluline"))
                .then(
                    pl.when(ORAL)
                    .then(pl.lit("blueline_oral→ETT"))
                    .otherwise(pl.lit("blueline_no_context"))
                )
                .otherwise(
                    pl.when(ORAL)
                    .then(pl.lit("ETT_brand+intubation"))
                    .otherwise(pl.lit("ETT_brand_no_context"))
                ),
            )
        )
        # 12. Nasopharyngeal / Wendl / Rachen
        .when(TYP.str.contains(NPT_STEMS) | (TYP == "wendel"))
        .then(classification_struct(CLASSES["NPT"], "very_high", "NPT/Wendl"))
        # 13. Oropharyngeal airway (incl. oral prosthetics Gaumenplatte, GaP)
        .when(
            TYP.str.contains(
                r"guedel|oropharyng|gaumenplatte|^gap\b|platte mit sporn|mundkeil"
            )
        )
        .then(classification_struct(CLASSES["OPA"], "high", "OPA"))
        # 14. Jet / transtracheal emergency
        .when(
            TYP.str.contains(
                r"\bjet\b|ravussin|cricath|quicktrach|quick trach|notfall-?koniotom"
            )
        )
        .then(classification_struct(CLASSES["JET"], "very_high", "jet/transtracheal"))
        # 15. Prongs: "Prong NIV" → NIV, otherwise → HFNC (default)
        .when(TYP.str.contains(r"\bprong") & TYP.str.contains(r"\bniv\b"))
        .then(classification_struct(CLASSES["NIV"], "very_high", "prong_NIV_explicit"))
        # 16. HFNC
        .when(TYP.str.contains(HFNC_STEMS))
        .then(classification_struct(CLASSES["HFNC"], "very_high", "HFNC"))
        .when(TYP.str.contains(r"\bprong|n-?\s?interf|nasenprongs"))
        .then(classification_struct(CLASSES["HFNC"], "very_high", "HFNC_prong_default"))
        # 17. O2 masks (low flow)
        .when(TYP.str.contains(O2M_STEMS))
        .then(classification_struct(CLASSES["O2M"], "very_high", "O2_mask"))
        # 18. O2 nasal cannula (low flow — "Brille" family)
        .when(TYP.str.contains(O2NC_STEMS))
        .then(
            classification_struct(
                CLASSES["O2NC"], "very_high", "O2_nasal_cannula_low_flow"
            )
        )  # 19. NIV masks (product family)
        # 19. NIV Family (Inner Branch)
        .when(TYP.str.contains(NIV_FAMILY))
        .then(
            classification_struct(
                cl=CLASSES["NIV"],
                conf=pl.when(
                    APP.str.contains(
                        r"facial|oronasal|full.?face|mund.?nase|nasal-oral|gesicht"
                    )
                )
                .then(pl.lit("very_high"))
                .when(APP.str.contains("nasal|binasal"))
                .then(pl.lit("high"))
                .otherwise(pl.lit("medium")),
                rule=pl.when(
                    APP.str.contains(
                        r"facial|oronasal|full.?face|mund.?nase|nasal-oral|gesicht"
                    )
                )
                .then(pl.lit("NIV_mask+face"))
                .when(APP.str.contains("nasal|binasal"))
                .then(pl.lit("NIV_nasal_mask"))
                .otherwise(pl.lit("NIV_mask_unspec_site")),
            )
        )
        # 20. Empty-typ fallbacks (context only)
        .when(TYP == "")
        .then(
            classification_struct(
                cl=pl.when(ORAL)
                .then(pl.lit(CLASSES["ETT"]))
                .when(TS)
                .then(pl.lit(CLASSES["TT"]))
                .when(APP.str.contains(r"facial|oronasal"))
                .then(pl.lit(CLASSES["NIV"]))
                .otherwise(pl.lit(CLASSES["UNCLEAR"])),
                conf="low",
                rule=pl.when(ORAL)
                .then(pl.lit("intubation_context_only"))
                .when(TS)
                .then(pl.lit("tracheotomy_context_only"))
                .when(APP.str.contains(r"facial|oronasal"))
                .then(pl.lit("facial_site_only"))
                .otherwise(pl.lit("no_rule_matched")),
            )
        )
        # Cuff-state markers without tube word
        .when(
            TYP.str.contains(
                r"^(gebl|ugebl|gebl sgla|gebl sgla ungef|gebl \(.*\)|gebl mrt-f)$"
            )
        )
        .then(
            classification_struct(
                cl=pl.when(ORAL)
                .then(pl.lit(CLASSES["ETT"]))
                .otherwise(pl.lit(CLASSES["UNCLEAR"])),
                conf=pl.when(ORAL).then(pl.lit("high")).otherwise(pl.lit("unclear")),
                rule=pl.when(ORAL)
                .then(pl.lit("gebl_oral→ETT"))
                .otherwise(pl.lit("gebl_no_context")),
            )
        )
        .otherwise(
            classification_struct(CLASSES["UNCLEAR"], "unclear", "no_rule_matched")
        )
    )

    # 3. Finalization
    subtype_expr = (
        pl.when(TYP.str.contains(r"doppellu|doppeltu|\bdlt\b"))
        .then(pl.lit("double_lumen"))
        .when(TYP.str.contains(r"elektrodentu|stimulationstubus|emg stim|medronic nro"))
        .then(pl.lit("EMG_electrode_tube"))
        .when(TYP.str.contains(r"bronchblock|bronchusblock|ez-block"))
        .then(pl.lit("with_bronchial_blocker"))
        .when(TYP.str.contains(r"\bsgla\b|\bsub.?glott"))
        .then(pl.lit("subglottic_suction"))
        .otherwise(None)
    )

    return (
        df.with_columns(res=expr)
        .with_columns(
            pl.col("res").struct.field("cl").alias("auto_class"),
            pl.col("res").struct.field("conf").alias("auto_conf"),
            pl.col("res").struct.field("rule").alias("auto_rule"),
            auto_subtype=subtype_expr,
        )
        .drop("__typ_n__", "__app_n__", "__anl_n__", "ts_ctx", "oral_ctx", "res")
    )
