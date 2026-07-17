"""Tests for idea_video.py — narrated slideshow generator."""
import json
import re
import pytest

from idea_video import (
    VIDEO_STYLES,
    generate_video_script,
    build_video_embed,
    build_video_html,
    estimate_duration_s,
)


_FULL_IDEA = {
    "title": "Graph Neural Networks for Drug Discovery",
    "method": ("Use message-passing neural networks on molecular graphs. "
                "Train on the ZINC benchmark. Evaluate on standard ADMET metrics."),
    "hypothesis": "GNNs outperform fingerprint baselines on drug screening tasks.",
    "expected_outcome": "15% AUROC improvement on standard benchmarks.",
    "methodology_type": "empirical_study",
    "quality_score": 0.78,
    "probe_scores": {
        "novelty": 0.7, "significance": 0.75, "clarity": 0.6,
        "testability": 0.65, "specificity": 0.5, "scalability": 0.55,
    },
}

_MINIMAL_IDEA = {"title": "Bare Idea"}


class TestVideoScript:
    def test_generates_at_least_six_slides(self):
        slides = generate_video_script(_FULL_IDEA)
        assert len(slides) >= 6

    def test_minimal_idea_still_produces_slides(self):
        slides = generate_video_script(_MINIMAL_IDEA)
        assert len(slides) >= 4
        for s in slides:
            assert s["title"] and s["narration"]

    def test_each_slide_has_required_fields(self):
        slides = generate_video_script(_FULL_IDEA)
        required = {"title", "subtitle", "body", "bullets", "narration",
                    "duration_s", "gradient", "icon"}
        for s in slides:
            assert required.issubset(s.keys())

    def test_first_slide_includes_title(self):
        slides = generate_video_script(_FULL_IDEA)
        assert _FULL_IDEA["title"] in slides[0]["title"] or \
               _FULL_IDEA["title"][:40] in slides[0]["narration"]

    def test_durations_are_positive_ints(self):
        slides = generate_video_script(_FULL_IDEA)
        for s in slides:
            assert isinstance(s["duration_s"], int) and s["duration_s"] > 0

    def test_estimate_duration(self):
        slides = generate_video_script(_FULL_IDEA)
        total = estimate_duration_s(slides)
        assert total == sum(s["duration_s"] for s in slides)
        assert 30 <= total <= 120  # Reasonable pitch length

    def test_narration_is_non_empty_strings(self):
        slides = generate_video_script(_FULL_IDEA)
        for s in slides:
            assert isinstance(s["narration"], str) and len(s["narration"]) > 5

    def test_method_split_into_bullets(self):
        slides = generate_video_script(_FULL_IDEA)
        method_slides = [s for s in slides if "How We" in s["title"]]
        assert method_slides, "Method slide missing"
        m = method_slides[0]
        # Method has 3 sentences → should produce bullets
        assert len(m["bullets"]) >= 1


class TestVideoStyles:
    def test_all_styles_have_required_metadata(self):
        required = {"label", "description", "default_rate",
                    "default_pitch", "captions"}
        for k, cfg in VIDEO_STYLES.items():
            assert required.issubset(cfg.keys()), f"{k} missing keys"

    def test_five_styles_exist(self):
        assert set(VIDEO_STYLES.keys()) == {
            "documentary", "trailer", "ted_talk", "news", "pitch_deck",
        }

    def test_each_style_generates_slides(self):
        for style in VIDEO_STYLES:
            slides = generate_video_script(_FULL_IDEA, style=style)
            assert len(slides) >= 4, f"{style} produced too few slides"
            for s in slides:
                assert s["title"] and s["narration"]
                assert s["duration_s"] > 0

    def test_styles_produce_distinct_scripts(self):
        # Different styles should produce different first-slide narration
        narrations = {
            style: generate_video_script(_FULL_IDEA, style=style)[0]["narration"]
            for style in VIDEO_STYLES
        }
        assert len(set(narrations.values())) == len(VIDEO_STYLES), \
            "All styles produced identical first slides"

    def test_unknown_style_falls_back_to_documentary(self):
        slides_unknown = generate_video_script(_FULL_IDEA, style="not_a_style")
        slides_doc = generate_video_script(_FULL_IDEA, style="documentary")
        assert len(slides_unknown) == len(slides_doc)
        assert slides_unknown[0]["narration"] == slides_doc[0]["narration"]

    def test_trailer_has_dramatic_hook(self):
        slides = generate_video_script(_FULL_IDEA, style="trailer")
        # Trailer-specific phrasing
        joined = " ".join(s["narration"].lower() for s in slides)
        assert "world" in joined or "summer" in joined or "coming" in joined

    def test_news_includes_lower_third(self):
        slides = generate_video_script(_FULL_IDEA, style="news")
        assert any("lower_third" in s for s in slides)
        joined = " ".join(s["narration"].lower() for s in slides)
        assert "breaking" in joined or "news" in joined

    def test_pitch_deck_has_ten_slides(self):
        slides = generate_video_script(_FULL_IDEA, style="pitch_deck")
        assert len(slides) == 10
        titles = [s["title"] for s in slides]
        assert "PROBLEM" in titles and "THE ASK" in titles

    def test_ted_talk_uses_personal_voice(self):
        slides = generate_video_script(_FULL_IDEA, style="ted_talk")
        joined = " ".join(s["narration"].lower() for s in slides)
        assert "i" in joined.split() or "thank you" in joined or "story" in joined


class TestVideoEmbed:
    def test_returns_html_string(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert isinstance(html, str) and len(html) > 1000

    def test_contains_player_root(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert 'id="ig-video-root"' in html

    def test_contains_all_slide_titles(self):
        import html as _h
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        for s in slides:
            escaped = _h.escape(s["title"])
            assert escaped in html or escaped[:20] in html

    def test_includes_play_button(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "ig-play" in html and "ig-mute" in html

    def test_includes_speech_synthesis_call(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "speechSynthesis" in html and "SpeechSynthesisUtterance" in html

    def test_slides_serialized_as_json(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        # Pull out the slides JSON and parse it
        m = re.search(r"const slides = (\[.*?\]);", html)
        assert m is not None, "Slide JSON not found"
        parsed = json.loads(m.group(1))
        assert len(parsed) == len(slides)

    def test_html_escapes_user_content(self):
        evil = {**_FULL_IDEA, "title": "<script>alert('xss')</script>",
                 "method": "<img src=x onerror=1>"}
        html = build_video_embed(generate_video_script(evil), evil)
        # Title should be HTML-escaped, not raw
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html or "&lt;script" in html

    def test_autoplay_flag_propagates(self):
        slides = generate_video_script(_FULL_IDEA)
        html_off = build_video_embed(slides, _FULL_IDEA, autoplay=False)
        html_on = build_video_embed(slides, _FULL_IDEA, autoplay=True)
        # Config object now wraps this — look for the autoplay flag in JSON
        assert '"autoplay": false' in html_off or '\\"autoplay\\": false' in html_off or '"autoplay":false' in html_off
        assert '"autoplay": true' in html_on or '"autoplay":true' in html_on

    def test_includes_voice_picker(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert 'class="ig-voice"' in html and "Auto" in html

    def test_includes_rate_slider(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert 'class="ig-rate"' in html
        assert 'type="range"' in html

    def test_includes_caption_container(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert 'class="ig-captions"' in html
        # Player JS handles word-boundary highlighting
        assert "onboundary" in html

    def test_includes_confetti_canvas(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert 'class="ig-confetti"' in html
        assert "fireConfetti" in html

    def test_style_class_applied_to_root(self):
        for style in VIDEO_STYLES:
            slides = generate_video_script(_FULL_IDEA, style=style)
            html = build_video_embed(slides, _FULL_IDEA, style=style)
            assert f"ig-style-{style}" in html

    def test_news_style_renders_lower_third(self):
        slides = generate_video_script(_FULL_IDEA, style="news")
        html = build_video_embed(slides, _FULL_IDEA, style="news")
        assert "ig-lower-third" in html
        assert "BREAKING" in html.upper()

    def test_default_rate_propagates_per_style(self):
        for style, cfg in VIDEO_STYLES.items():
            slides = generate_video_script(_FULL_IDEA, style=style)
            html = build_video_embed(slides, _FULL_IDEA, style=style)
            # The rate value appears in the slider's value attribute
            rate_str = f'value="{cfg["default_rate"]}"'
            assert rate_str in html, f"{style} default_rate not in slider"

    def test_invalid_style_falls_back_in_embed(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA, style="not_a_style")
        # Should still render (fallback to documentary)
        assert "ig-style-documentary" in html


class TestDataVisuals:
    def test_documentary_has_visuals_on_data_slides(self):
        slides = generate_video_script(_FULL_IDEA, style="documentary")
        with_viz = [s for s in slides if s.get("data_visual")]
        assert len(with_viz) >= 3, "Expected visuals on multiple slides"

    def test_visual_types_cover_all_kinds(self):
        # Across all styles we should see every visual type
        seen = set()
        for style in VIDEO_STYLES:
            for s in generate_video_script(_FULL_IDEA, style=style):
                if s.get("data_visual"):
                    seen.add(s["data_visual"]["type"])
        assert {"bars", "gauge", "counter", "histogram", "timeline"}.issubset(seen)

    def test_bars_visual_has_items(self):
        slides = generate_video_script(_FULL_IDEA, style="documentary")
        bars = [s["data_visual"] for s in slides
                 if s.get("data_visual", {}).get("type") == "bars"]
        assert bars
        for b in bars:
            assert isinstance(b["items"], list) and len(b["items"]) >= 1
            for it in b["items"]:
                assert "label" in it and "value" in it
                assert 0 <= float(it["value"]) <= 100

    def test_gauge_value_in_range(self):
        for style in VIDEO_STYLES:
            for s in generate_video_script(_FULL_IDEA, style=style):
                v = s.get("data_visual")
                if v and v["type"] == "gauge":
                    assert 0 <= float(v["value"]) <= 100

    def test_pitch_deck_has_timeline(self):
        slides = generate_video_script(_FULL_IDEA, style="pitch_deck")
        tls = [s for s in slides
               if s.get("data_visual", {}).get("type") == "timeline"]
        assert len(tls) == 1
        assert len(tls[0]["data_visual"]["milestones"]) == 3

    def test_visual_html_renders_for_each_type(self):
        slides = generate_video_script(_FULL_IDEA, style="documentary")
        html = build_video_embed(slides, _FULL_IDEA, style="documentary")
        # Every viz class should appear at least once
        for cls in ("ig-bars-viz", "ig-gauge-viz",
                    "ig-counter-viz", "ig-histogram-viz"):
            assert cls in html, f"Missing CSS class {cls} in rendered embed"

    def test_pitch_deck_timeline_renders(self):
        slides = generate_video_script(_FULL_IDEA, style="pitch_deck")
        html = build_video_embed(slides, _FULL_IDEA, style="pitch_deck")
        assert "ig-timeline-viz" in html and "ig-tl-dot" in html


class TestVoiceQualityAndCadence:
    def test_voice_scoring_function_present(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "scoreVoice" in html and "pickBestVoice" in html

    def test_voice_scoring_rewards_natural(self):
        # Verify the actual scoring logic by extracting and inspecting
        # the regex patterns in the JS — ensures "Natural"/"Neural" voices
        # win over robotic ones
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "natural|neural" in html.lower() or "neural" in html
        # eSpeak should be penalized
        assert "espeak" in html.lower() or "robot" in html.lower()

    def test_sentence_cadence_engine_present(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "speakSentences" in html
        assert "splitSentences" in html
        # Pitch/rate variation present (the lines that randomize)
        assert "Math.random" in html

    def test_natural_pause_lengths_in_cadence(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        # Distinct pause durations for ?/!, ., comma, and clause breaks
        assert "320" in html and "200" in html and "120" in html

    def test_speaker_indicator_in_embed(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "ig-speaker" in html and "ig-speaker-dot" in html

    def test_voice_picker_uses_quality_badges(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        # Auto-best label includes a sparkle for high quality
        assert "✨" in html or "Auto" in html


class TestVisualAnimation:
    def test_animate_function_present(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_embed(slides, _FULL_IDEA)
        assert "animateSlideVisual" in html

    def test_visual_data_attributes_present(self):
        slides = generate_video_script(_FULL_IDEA, style="documentary")
        html = build_video_embed(slides, _FULL_IDEA, style="documentary")
        # data-target attributes drive the animation
        assert 'data-target="' in html
        assert 'data-type="bars"' in html or 'data-type="gauge"' in html


class TestDialogueMode:
    def test_speak_sentences_dialogue_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "speakSentencesDialogue" in html

    def test_voice_pair_picker_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "pickPair" in html and "classifyGender" in html

    def test_pair_picker_handles_gender_names(self):
        # Verify the female/male regex covers a sensible spread of names
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        for name in ("aria", "samantha", "jenny"):
            assert name in html.lower()
        for name in ("guy", "daniel", "alex"):
            assert name in html.lower()

    def test_mode_toggle_button_in_embed(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-mode-toggle" in html and "Solo" in html

    def test_dialogue_speaker_styling(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-cap-a" in html and "ig-cap-b" in html
        assert "data-host" in html or "[data-host" in html

    def test_pitch_modulation_for_single_voice_fallback(self):
        # When only one voice is available, the dialogue still differentiates
        # speakers via pitch (Host A high, Host B low)
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "1.18" in html and "0.82" in html  # the fallback pitch modifiers


class TestChapterMarkers:
    def test_renderChapters_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "renderChapters" in html and "ig-chapters" in html

    def test_chapters_call_show_on_click(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Click handler invokes show(i)
        assert "ig-chapter-tip" in html

    def test_chapter_container_inside_progress_track(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # The container precedes any later content, ordering check
        track_idx = html.find('class="ig-progress-track"')
        chapt_idx = html.find('class="ig-chapters"')
        # Both present and chapters comes after track open
        assert track_idx >= 0 and chapt_idx > track_idx


class TestParticleBackgrounds:
    def test_each_style_has_animation_keyframes(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # All animation names should appear in the bundled CSS
        for kf in ("ig-drift", "ig-embers", "ig-glow",
                   "ig-scan", "ig-grid-pulse"):
            assert kf in html, f"keyframes {kf} missing"

    def test_documentary_style_uses_drift(self):
        html = build_video_embed(
            generate_video_script(_FULL_IDEA, style="documentary"),
            _FULL_IDEA, style="documentary",
        )
        assert "ig-drift" in html and "ig-style-documentary" in html

    def test_trailer_style_uses_embers(self):
        html = build_video_embed(
            generate_video_script(_FULL_IDEA, style="trailer"),
            _FULL_IDEA, style="trailer",
        )
        assert "ig-embers" in html

    def test_news_style_uses_scan(self):
        html = build_video_embed(
            generate_video_script(_FULL_IDEA, style="news"),
            _FULL_IDEA, style="news",
        )
        assert "ig-scan" in html

    def test_pitch_deck_uses_grid(self):
        html = build_video_embed(
            generate_video_script(_FULL_IDEA, style="pitch_deck"),
            _FULL_IDEA, style="pitch_deck",
        )
        assert "ig-grid-pulse" in html


class TestSynthesizedMusic:
    def test_music_toggle_in_embed(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-music-toggle" in html and "🎵 Music off" in html

    def test_music_engine_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "startMusic" in html and "stopMusic" in html
        assert "AudioContext" in html

    def test_music_configs_for_all_styles(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "MUSIC_CONFIGS" in html
        for style in VIDEO_STYLES:
            assert style in html  # At minimum the style key appears in config

    def test_uses_lowpass_filter(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Filter and LFO modulation evidence
        assert "createBiquadFilter" in html or "lowpass" in html
        assert "createOscillator" in html

    def test_music_starts_on_play_when_enabled(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Confirm music start is gated on the user toggle
        assert "musicEnabled" in html and "startMusic()" in html


class TestEmojiReactions:
    def test_reaction_rules_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "REACTION_RULES" in html and "detectReactions" in html

    def test_reactions_cover_positive_negative_neutral(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Spot-check across categories
        for emoji in ("✨", "💡", "🚀", "⚠️", "🤔", "🏆"):
            assert emoji in html, f"Missing reaction emoji {emoji}"

    def test_reactions_fire_for_both_solo_and_dialogue(self):
        # detectReactions should be invoked from both engines
        # (solo engine calls with `phrase`, dialogue engine calls with `sent`)
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        total = (html.count("detectReactions(phrase)")
                 + html.count("detectReactions(sent)"))
        assert total >= 2

    def test_emit_reaction_appends_to_stage(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "emitReaction" in html and "ig-reaction" in html


class TestContentReveal:
    def test_reveal_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "revealSlideContent" in html

    def test_typewriter_class(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-typing" in html

    def test_bullets_stagger_uses_hidden_class(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-hidden" in html

    def test_reveal_called_from_show(self):
        # show() must invoke revealSlideContent for the reveal to fire on
        # every slide change (not just the first)
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "revealSlideContent(slideEls[idx])" in html


class TestHumanization:
    def test_humanize_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function humanize" in html
        assert "HUMAN_HOOKS" in html and "MID_FILLERS" in html

    def test_style_specific_hooks_for_each_style(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Each of the five styles has its own hook bank
        for style in VIDEO_STYLES:
            # Hook bank uses the style key as a property name
            assert style + ":" in html or style + " :" in html

    def test_formal_to_casual_replacements(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Verify a few of the formal→casual rules made it into the JS
        assert "FORMAL_TO_CASUAL" in html
        assert "utilize" in html.lower() and "nonetheless" in html.lower()
        assert "in conclusion" in html.lower()

    def test_humanize_called_in_solo_engine(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Both engines must humanize their input text
        assert "humanize(rawText)" in html
        # Solo engine accepts rawText param
        assert "function speakSentences(rawText)" in html

    def test_humanize_called_in_dialogue_engine(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function speakSentencesDialogue(rawText)" in html

    def test_humanize_toggle_button_in_embed(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "ig-human-toggle" in html
        assert "🧑 Human" in html


class TestPhraseAndArc:
    def test_split_phrases_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "splitPhrases" in html

    def test_phrase_splitter_handles_internal_punct(self):
        # The splitter regex covers commas, semicolons, em/en dashes
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Look for the regex characters in source
        assert "[,;—–]" in html or "[,;\\u2014\\u2013]" in html

    def test_rate_arc_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function rateArc" in html
        # Sine-shaped arc invokes Math.sin and Math.PI
        assert "Math.sin" in html and "Math.PI" in html

    def test_rate_arc_used_in_engines(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Solo + Dialogue both call rateArc on each phrase/sentence
        assert html.count("rateArc(") >= 2


class TestBreathAndBackchannels:
    def test_play_breath_function(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function playBreath" in html
        # Uses bandpass filtered noise burst
        assert "bandpass" in html and "createBuffer" in html

    def test_breath_fires_in_solo_engine(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Breath called probabilistically inside utterance.onend
        assert "playBreath()" in html

    def test_backchannels_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "BACKCHANNELS" in html and "playBackchannel" in html
        for word in ("Mm-hmm", "Right", "Yeah"):
            assert word in html, f"Missing backchannel: {word}"

    def test_backchannel_uses_other_voice(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Listener (otherVoice) speaks the backchannel
        assert "otherVoice" in html and "playBackchannel(otherVoice)" in html

    def test_backchannel_volume_lower(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Backchannel utterance volume is reduced (not 1.0)
        assert "u.volume = 0.55" in html


class TestEmotionProsody:
    def test_emotion_classifier_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "classifyEmotion" in html and "EMOTION_PROSODY" in html

    def test_all_emotion_categories_defined(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        for emo in ("question", "excitement", "concern", "curious", "soft", "neutral"):
            assert emo in html, f"Missing emotion category: {emo}"

    def test_question_pitch_higher_than_neutral(self):
        # Pitch values for the question prosody should be > 1.0 (rising)
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "pitch: 1.12" in html  # question
        assert "pitch: 1.10" in html  # excitement
        assert "pitch: 0.88" in html  # concern (lower)

    def test_emotion_prosody_used_in_solo_engine(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "classifyEmotion(phrase)" in html
        assert "prosody.pitch" in html and "prosody.rate" in html

    def test_emotion_prosody_used_in_dialogue_engine(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "classifyEmotion(sent)" in html


class TestDrawnOutFillers:
    def test_filler_detection_function(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "isFillerPhrase" in html

    def test_filler_keywords_covered(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # The filler regex includes the common ones
        for word in ("um", "uh", "hmm", "so", "well", "right"):
            # They appear in the regex source
            assert word in html

    def test_filler_rate_slowdown_applied(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Filler delivery uses 0.72 rate multiplier
        assert "arcRate * 0.72" in html

    def test_filler_volume_reduced(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Fillers play at 0.85 volume so they sit beneath the main delivery
        assert "filler ? 0.85" in html


class TestLipSmackAndDucking:
    def test_lip_smack_function_present(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function playLipSmack" in html
        # Highpass filter for the click character
        assert "highpass" in html

    def test_lip_smack_called_on_first_phrase(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "playLipSmack()" in html

    def test_music_ducking_functions(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "function duckMusic" in html
        assert "function unduckMusic" in html
        assert "musicDucked" in html

    def test_duck_music_called_on_speech_start(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # u.onstart calls duckMusic in the cadence engines
        assert "duckMusic()" in html
        assert "unduckMusic()" in html

    def test_duck_target_is_30_percent(self):
        html = build_video_embed(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        # Duck multiplier 0.30 is what makes the music sit under the voice
        assert "0.30" in html


class TestVideoFullHTML:
    def test_is_full_document(self):
        slides = generate_video_script(_FULL_IDEA)
        html = build_video_html(slides, _FULL_IDEA)
        assert html.startswith("<!DOCTYPE html>") or html.startswith("<!doctype")
        assert "</html>" in html

    def test_includes_viewport_meta(self):
        html = build_video_html(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert 'name="viewport"' in html

    def test_title_in_head(self):
        html = build_video_html(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert "<title>" in html and "IdeaGraph" in html

    def test_contains_player_embed(self):
        html = build_video_html(generate_video_script(_FULL_IDEA), _FULL_IDEA)
        assert 'id="ig-video-root"' in html
