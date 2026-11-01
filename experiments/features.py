"""Feature sets built on the old make_features. build() counts over train+test features; reference_features() can use any reference."""

import numpy as np
import pandas as pd

from common import make_features

AUDIO = ["danceability", "energy", "loudness", "acousticness", "instrumentalness",
         "speechiness", "valence", "liveness", "tempo", "duration_ms"]


def audio_key(both):
    key = pd.DataFrame({
        "dur": (both.duration_ms / 2000).round(),  # shortcut: 2 s buckets miss pairs straddling a boundary, fine for a count feature
        "tempo": both.tempo.round(),
        "energy": (both.energy * 50).round(),
        "dance": (both.danceability * 50).round(),
    }).astype(str).agg("|".join, axis=1)
    key[both[["duration_ms", "tempo"]].isna().any(axis=1)] = np.nan
    return key


def pct_in_group(ref_values, ref_groups, values, groups):
    """Fraction of reference values in the same group that are <= each value."""
    out = np.full(len(values), np.nan)
    for name, vals in pd.Series(np.asarray(ref_values)).groupby(np.asarray(ref_groups)):
        ordered, mask = np.sort(vals.dropna()), np.asarray(groups) == name
        out[mask] = np.searchsorted(ordered, np.asarray(values)[mask], side="right") / len(ordered)
    return out


def engineered(ref, query, f, variant):
    """Round-10 features from the brainstorm's single-feature AUC checks. variant: eng_vocal/era/dance/misc or eng (all)."""
    q = query.reset_index(drop=True)
    if variant in ("eng_vocal", "eng"):
        f["vocal_presence"] = (1 - q.instrumentalness) - q.speechiness
        f["no_instrumental"] = (q.instrumentalness == 0).astype(int)
    if variant in ("eng_era", "eng"):
        # Energy/valence point to High after 2023 but flip in 2015-19, so give each era its own copy.
        year = f.release_year
        eras = {"pre2015": year < 2015, "y2015_19": year.between(2015, 2019), "y2020_22": year.between(2020, 2022), "y2023": year >= 2023}
        slope = np.polyfit(ref.loudness.fillna(ref.loudness.median()), ref.energy.fillna(ref.energy.median()), 1)
        energy_resid = q.energy - np.polyval(slope, q.loudness)
        for name, mask in eras.items():
            for col, values in (("energy", q.energy), ("valence", q.valence), ("speechiness", q.speechiness), ("energy_resid", energy_resid)):
                f[f"{col}_{name}"] = values.where(mask, 0)
    if variant in ("eng_dance", "eng"):
        f["dance_pct_in_year"] = pct_in_group(ref.danceability, make_features(ref).release_year, q.danceability, f.release_year)
    if variant in ("eng_misc", "eng"):
        f["calm_acoustic"] = ((q.acousticness > 0.6) & (q.energy < 0.4)).astype(int)
        full_date = lambda d: pd.to_datetime(d.where(d.str.len() == 10), errors="coerce")
        months = (full_date(ref.track_album_release_date).max() - full_date(q.track_album_release_date)).dt.days / 30.4
        f["months_before_snapshot"] = months.where(f.release_year >= 2023)  # snapshot = latest train release
    return f


def reference_features(ref, query, query_is_ref, variant="all"):
    """Features for `query` rows using only `ref` (training) rows as the reference; query rows never see each other.

    query_is_ref: the query rows are the reference rows themselves (training features), so they aren't counted twice.
    """
    f = make_features(query).reset_index(drop=True)
    eng, variant = variant, ("all" if variant.startswith("eng") else variant)
    extra = 0 if query_is_ref else 1  # a query row outside the reference counts itself once
    if variant in ("dup", "dup_near", "all"):
        f["uri_count"] = np.maximum(query.uri.map(ref.uri.value_counts()).fillna(0).to_numpy() + extra, 1)  # missing uri counts as itself
    if variant in ("dup_near", "all"):
        # Re-releases: other tracks in the reference with near-identical audio.
        ref_keys = pd.DataFrame({"key": audio_key(ref), "uri": ref.uri}).drop_duplicates()
        q = pd.DataFrame({"key": audio_key(query).to_numpy(), "uri": query.uri.to_numpy()})
        own_in_ref = q.set_index(["key", "uri"]).index.isin(ref_keys.set_index(["key", "uri"]).index)
        f["near_dup_count"] = np.maximum(q.key.map(ref_keys.key.value_counts()).fillna(0).to_numpy() - own_in_ref, 0)  # missing audio key
    if variant in ("yearrel", "all"):
        # Loudness war etc.: how a track compares with its release-year peers.
        ref_year = make_features(ref).release_year.clip(lower=1990).to_numpy()
        for col in ["loudness", "energy", "acousticness", "danceability", "duration_ms", "instrumentalness"]:
            medians = ref[col].groupby(ref_year).median()
            f[col + "_vs_year"] = query[col].to_numpy() - f.release_year.clip(lower=1990).map(medians).fillna(ref[col].median()).to_numpy()
    if variant in ("date_count", "all"):
        f["date_count"] = query.track_album_release_date.map(ref.track_album_release_date.value_counts()).fillna(0).to_numpy() + extra
    if eng.startswith("eng"):
        f = engineered(ref, query, f, eng)
    return f


def build(train, test, variant):
    """Train and test features with every count/median taken over train+test features (never labels)."""
    both = pd.concat([train.drop(columns="Popularity_Type"), test.drop(columns="ID")], ignore_index=True)
    f = reference_features(both, both, True, variant)
    return f.iloc[:len(train)].reset_index(drop=True), f.iloc[len(train):].reset_index(drop=True)


if __name__ == "__main__":
    from common import load
    train, test, y, _ = load()
    tr, te = build(train, test, "all")
    assert len(tr) == len(train) and len(te) == len(test)
    assert tr.uri_count.min() >= 1 and (tr.near_dup_count >= 0).all()
    assert (te.uri_count > 1).sum() == 22  # 11 tracks repeat within test
    ref = train.drop(columns="Popularity_Type").head(6).copy()
    ref.loc[1, "uri"] = ref.loc[0, "uri"]
    assert reference_features(ref, ref, True).uri_count.tolist()[:3] == [2, 2, 1]
    assert reference_features(ref, ref.iloc[[0]], False).uri_count.iloc[0] == 3  # 2 in ref + itself
    assert reference_features(ref, ref.iloc[[2]].assign(uri="new"), False).near_dup_count.iloc[0] == 1
    for v in ("eng_vocal", "eng_era", "eng_dance", "eng_misc", "eng"):
        a, b = build(train, test, v)
        assert list(a.columns) == list(b.columns) and len(a) == len(train) and len(b) == len(test)
    print("features self-check ok")
