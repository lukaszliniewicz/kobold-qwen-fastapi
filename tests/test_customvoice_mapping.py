from main import (
    PRESET_VOICE_DATA,
    customvoice_presets_ready,
    resolve_preset_voice,
)


def test_every_preset_id_and_label_resolves_to_its_canonical_speaker():
    for preset in PRESET_VOICE_DATA:
        assert resolve_preset_voice(preset["id"])["id"] == preset["id"]
        assert resolve_preset_voice(preset["name"])["id"] == preset["id"]
        assert resolve_preset_voice(preset["id"].swapcase())["id"] == preset["id"]


def test_customvoice_readiness_requires_all_nine_patched_speakers():
    speakers = [preset["id"].lower() for preset in PRESET_VOICE_DATA]
    assert customvoice_presets_ready(speakers, "qwen3tts-customvoice-v1")
    assert not customvoice_presets_ready(speakers, "")
    assert not customvoice_presets_ready(speakers[:-1], "qwen3tts-customvoice-v1")
    assert not customvoice_presets_ready(
        ["kobo", "cheery", "sleepy", "shouty", "chatty"],
        "qwen3tts-customvoice-v1",
    )
