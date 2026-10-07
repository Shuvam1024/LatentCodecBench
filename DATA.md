# Data

The clip list is [`manifests/clips.yaml`](manifests/clips.yaml). Video bytes are not in this repository. `lcb data fetch` downloads each source, checks the SHA256 recorded in the manifest, and writes `data/clips/{id}.y4m`. A file that does not match is deleted.

The canonical picture is **832×480, 33 frames, 8-bit 4:2:0 y4m**, tagged BT.709. Why that shape, and why 34 clips, is [ADR-0002](docs/decisions/0002-canonical-clips.md).

## What the SHA256 covers

The digest is of the bytes the fetcher keeps, not necessarily of a whole published object.

- **Xiph derf.** The manifest sets `byte_length`. That is an HTTP range: the y4m header plus 33 frames. The fetcher requires HTTP 206 and rejects a body of any other length. Re-hashing the full multi-gigabyte y4m is a different check, and this file does not claim it.
- **UVG and the camera clips.** The digest is of the whole downloaded file. The three UVG digests are the SHA256SUMS published beside the Xiph UVG mirror (`https://media.xiph.org/uvg/`).

## Licenses

### Xiph derf HD

Source page: <https://media.xiph.org/video/derf/>

That page says the sequences are believed to be freely redistributable for research, and that some of them may carry extra restrictions. The manifest records that statement as the license for the derf clips. It does not invent a tighter grant.

Two exceptions are called out on the clip itself:

- **Sintel trailer** is the Blender Foundation film, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). It is animation, not a camera capture.
- **FourPeople, Johnny, KristenAndSara** are the HEVC common-test-condition conference sequences, mirrored by Xiph. Treat them as research material under the Xiph redistribution note.

Derf files are raw y4m. A few are 4:2:2; preprocess converts those to 4:2:0. That is a chroma subsample, not a codec. These clips are not already compressed.

### UVG

ShakeNDry, Bosphorus, and HoneyBee. Each archive contains a copyright note from Digiturk:

> This sequence and all intellectual property rights therein remain the property of Digiturk. You may use this sequence according to Creative Commons Attribution-NonCommercial 3.0 Unported.

License: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/).

The archive names say `120fps`. The y4m header inside each file is `F30:1`, 3840×2160, 10-bit 4:2:0. The manifest records 30/1, which is the rate ffmpeg sees. Preprocess converts 10-bit to 8-bit 4:2:0 and stops after 33 frames, streamed out of the 7z member so the multi-gigabyte y4m is never written to disk. These three are uncompressed samples. They are not already codec-compressed.

The other UVG archives are the same kind of 4K material. Three sequences cover the slow-texture, water, and camera-motion end of that set. See ADR-0002.

### Camera clips

These four are short Creative Commons videos from Wikimedia Commons. Each one is **already compressed**. The canonical y4m is a decode of that bitstream, so a classic-codec baseline on these clips is a re-encode. Derf and UVG are not in that category. Phase 3 (codec leak) is why the distinction is recorded here.

| id | file | codec already on the source | native | license |
| --- | --- | --- | --- | --- |
| `street_50` | [50 Street video](https://commons.wikimedia.org/wiki/File:50_Street_video_vc.ogv) | Theora | 1440×1080, 25/1 | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| `lawn_mower` | [Handheld lawn mower](https://commons.wikimedia.org/wiki/File:Handheld_lawn_mower.webm) | VP9 | 1080×1920 (portrait), 30/1 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `rust_laser` | [Cleaning rust with a handheld laser](https://commons.wikimedia.org/wiki/File:Cleaning_rust_with_handheld_laser.webm) | VP9 | 854×480, 25/1 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `volcano` | [Litli-Hrútur eruption, Giles Laurent, 2023](https://commons.wikimedia.org/wiki/File:007_Volcano_eruption_of_Litli-Hr%C3%BAtur_in_Iceland_in_2023_Video_by_Giles_Laurent.webm) | VP9 | 3840×2160, 30000/1001 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Portrait footage is cover-then-center-cropped to 832×480. The sides of the frame are what remains; the top and bottom are cut.

## The 34 clips

| id | category | native | fps | hash covers |
| --- | --- | --- | --- | --- |
| crowd_run | high-motion | 1920×1080 | 50/1 | 33-frame range |
| park_joy | high-motion | 1920×1080 | 50/1 | 33-frame range |
| ducks_take_off | water | 1920×1080 | 50/1 | 33-frame range |
| in_to_tree | foliage | 1920×1080 | 50/1 | 33-frame range |
| old_town_cross | camera-pan | 1920×1080 | 50/1 | 33-frame range |
| pedestrian_area | people | 1920×1080 | 25/1 | 33-frame range |
| riverbed | water | 1920×1080 | 25/1 | 33-frame range |
| rush_hour | traffic | 1920×1080 | 25/1 | 33-frame range |
| sunflower | fine-detail | 1920×1080 | 25/1 | 33-frame range |
| tractor | camera-motion | 1920×1080 | 25/1 | 33-frame range |
| blue_sky | sky | 1920×1080 | 25/1 | 33-frame range |
| station2 | urban | 1920×1080 | 25/1 | 33-frame range |
| touchdown_pass | sports | 1920×1080 | 30000/1001 | 33-frame range |
| speed_bag | periodic-motion | 1920×1080 | 30000/1001 | 33-frame range |
| controlled_burn | smoke | 1920×1080 | 30000/1001 | 33-frame range |
| red_kayak | water-sports | 1920×1080 | 30000/1001 | 33-frame range |
| dinner | indoor | 1920×1080 | 30/1 | 33-frame range |
| factory | industrial | 1920×1080 | 30/1 | 33-frame range |
| aspen | wind | 1920×1080 | 30000/1001 | 33-frame range |
| snow_mnt | landscape | 1920×1080 | 30000/1001 | 33-frame range |
| west_wind_easy | wind | 1920×1080 | 30000/1001 | 33-frame range |
| rush_field_cuts | scene-cuts | 1920×1080 | 30000/1001 | 33-frame range |
| life | nature | 1920×1080 | 30/1 | 33-frame range |
| four_people | conference | 1280×720 | 60/1 | 33-frame range |
| johnny | conference | 1280×720 | 60/1 | 33-frame range |
| kristen_and_sara | conference | 1280×720 | 60/1 | 33-frame range |
| sintel_trailer | animation | 1920×1080 | 24/1 | 33-frame range |
| shaken_dry | water | 3840×2160 | 30/1 | whole 7z |
| bosphorus | camera-motion | 3840×2160 | 30/1 | whole 7z |
| honey_bee | fine-detail | 3840×2160 | 30/1 | whole 7z |
| street_50 | urban | 1440×1080 | 25/1 | whole file |
| lawn_mower | camera-motion | 1080×1920 | 30/1 | whole file |
| rust_laser | fine-detail | 854×480 | 25/1 | whole file |
| volcano | landscape | 3840×2160 | 30000/1001 | whole file |
