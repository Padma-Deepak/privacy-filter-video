"""Draw-to-keep behavior, rendering, session ownership and original cleanup."""
import io
import json
import sys
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'project'))
from core import review
from core.profiles import load_profile
from core.selection import match_face_selection, validate_choices, validate_manual, is_hidden
from review_api import register_review


def record(key, box, cls='faces', source='detected'):
    return {'id': key, 'class': cls, 'box': box, 'raw_box': box, 'source': source}


def test_draw_matches_only_the_selected_face():
    frame = [record('faces:1', [10, 10, 20, 20]), record('faces:2', [60, 10, 20, 20]),
             record('plates:1', [10, 10, 20, 20], 'plates')]
    assert match_face_selection([8, 8, 24, 24], frame) == 'faces:1'


@pytest.mark.parametrize('frame', [[], [record('faces:1', [10, 10, 20, 20], source='gap_fill')],
    [record('faces:1', [10, 10, 20, 20]), record('faces:2', [11, 10, 20, 20])]])
def test_missing_gap_only_or_ambiguous_face_fails_closed(frame):
    with pytest.raises(ValueError):
        match_face_selection([8, 8, 25, 25], frame)


def test_selection_does_not_attach_to_giant_false_detection():
    with pytest.raises(ValueError):
        match_face_selection([40, 40, 20, 20], [record('faces:1', [0, 0, 100, 100])])


def test_keep_and_range_choices_and_unknown_ids():
    choices = validate_choices({'faces:1': {'mode': 'keep'},
                               'faces:2': {'mode': 'range', 'start': 1, 'end': 2}},
                              {'faces:1': {}, 'faces:2': {}}, 4)
    assert not is_hidden('faces:1', 0, choices)
    assert is_hidden('faces:2', 1, choices)
    assert not is_hidden('faces:2', 3, choices)
    assert is_hidden('faces:3', 0, choices)
    with pytest.raises(ValueError):
        validate_choices({'typo': {'mode': 'keep'}}, {}, 1)
    with pytest.raises(ValueError):
        validate_manual([{'box': [0, 0, 1, 1], 'start': -1, 'end': 0}], 10, 10, 1)


@pytest.fixture
def client_app(tmp_path, monkeypatch):
    def fake_detect(frame, models, profile):
        return {'faces': [(10, 10, 20, 20), (60, 10, 20, 20)]}
    monkeypatch.setattr(review, 'detect', fake_detect)
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'project/templates'))
    app.secret_key = 'synthetic-test-only'
    app.config['TESTING'] = True
    register_review(app, str(tmp_path), {'png'}, {'mp4'})
    client = app.test_client()
    client.get('/review')
    with client.session_transaction() as session:
        token = session['csrf']
    yield app, client, {'X-CSRF-Token': token}
    app.extensions['review_jobs'].close()


def wait_ready(client, key, target='ready'):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = client.get(f'/api/jobs/{key}').json
        if state['state'] == target:
            return state
        if state['state'] == 'error':
            pytest.fail(state['error'])
        time.sleep(.02)
    pytest.fail('Background job timed out')


def create_image(client, headers):
    source = np.random.default_rng(42).integers(40, 255, (64, 100, 3), dtype=np.uint8)
    _, png = cv2.imencode('.png', source)
    response = client.post('/api/jobs', data={'file': (io.BytesIO(png.tobytes()), 'image.png'),
        'profile': 'journalist', 'faces_only': 'true'}, headers=headers)
    assert response.status_code == 202
    key = response.json['id']
    wait_ready(client, key)
    return key, source


def test_draw_keep_export_and_source_deletion(client_app):
    app, client, headers = client_app
    key, original = create_image(client, headers)
    selection = client.post(f'/api/jobs/{key}/select-face', json={'frame': 0, 'box': [8, 8, 24, 24]}, headers=headers)
    assert selection.status_code == 200
    assert selection.json['track']['id'] == 'faces:1'
    assert selection.json['positions'][0]['box'] == [10, 10, 20, 20]
    export = client.post(f'/api/jobs/{key}/export', json={'choices': {'faces:1': {'mode': 'keep'}}}, headers=headers)
    assert export.status_code == 202
    wait_ready(client, key, 'complete')
    output = client.get(f'/api/jobs/{key}/download')
    image = cv2.imdecode(np.frombuffer(output.data, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(image[10:30, 10:30], original[10:30, 10:30])
    assert image[10:30, 60:80].max() == 0
    job = app.extensions['review_jobs'].jobs[key]
    assert not job.source.exists() and job.analysis is None
    report = client.get(f'/api/jobs/{key}/report').json
    assert report['tracks'][0]['hidden_frames'] == 0
    assert report['tracks'][1]['hidden_frames'] == 1
    assert 'thumbnail' not in json.dumps(report)
    assert client.get(f'/api/jobs/{key}/frames/0').status_code == 404
    assert client.get(f'/api/jobs/{key}/report?format=html').status_code == 200


def test_multiple_kept_people_and_manual_hide_priority():
    frame = np.full((64, 100, 3), 200, np.uint8)
    data = {'width': 100, 'height': 64, 'frames': [[record('faces:1', [10,10,20,20]), record('faces:2',[60,10,20,20])]]}
    kept = {'faces:1': {'mode': 'keep'}, 'faces:2': {'mode': 'keep'}}
    review.render_frame(frame, 0, data, kept, [{'box':[10,10,10,10], 'start':0,'end':0}], load_profile('journalist'))
    assert frame[10:20,10:20].max() == 0
    assert frame[10:30,60:80].min() == 200


def test_csrf_and_cross_session_isolation(client_app):
    app, client, headers = client_app
    key, _ = create_image(client, headers)
    other = app.test_client()
    assert other.get(f'/api/jobs/{key}').status_code == 404
    assert other.get(f'/api/jobs/{key}/source').status_code == 404
    assert client.post(f'/api/jobs/{key}/select-face', json={}).status_code == 403
    assert client.post(f'/api/jobs/{key}/select-face', json={'frame':0,'box':[-1,0,10,10]}, headers=headers).status_code == 400
    assert client.get(f'/api/jobs/{key}').headers['Cache-Control'] == 'no-store, private'


def test_delete_ready_job_removes_upload(client_app):
    app, client, headers = client_app
    key, _ = create_image(client, headers)
    job = app.extensions['review_jobs'].jobs[key]
    assert job.source.exists()
    assert client.delete(f'/api/jobs/{key}', headers=headers).status_code == 202
    assert not job.directory.exists()
    assert job.analysis is None


@pytest.mark.parametrize('audio', ['keep','mute'])
def test_video_playback_range_and_export_audio_metadata(client_app, tmp_path, audio):
    app, client, headers = client_app
    source = tmp_path / 'synthetic.mp4'
    subprocess.run(['ffmpeg','-y','-f','lavfi','-i','testsrc=size=100x64:rate=5:duration=1',
        '-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-pix_fmt','yuv420p',
        '-c:a','aac','-metadata','location=+12.345+045.678/','-metadata','title=private-source',
        str(source),'-loglevel','error'],check=True)
    response=client.post('/api/jobs',data={'file':(io.BytesIO(source.read_bytes()),'clip.mp4'),
        'profile':'journalist','faces_only':'true'},headers=headers)
    key=response.json['id'];wait_ready(client,key)
    media=client.get(f'/api/jobs/{key}/source',headers={'Range':'bytes=0-99'})
    assert media.status_code==206 and len(media.data)==100
    assert client.post(f'/api/jobs/{key}/export',json={'audio':audio,'choices':{'faces:1':{'mode':'keep'}}},headers=headers).status_code==202
    wait_ready(client,key,'complete')
    output=app.extensions['review_jobs'].jobs[key].output
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
    assert any(s['codec_name']=='h264' for s in probe['streams'])
    assert any(s['codec_type']=='audio' for s in probe['streams']) == (audio=='keep')
    assert 'private-source' not in json.dumps(probe) and '+12.345' not in json.dumps(probe)
    assert not app.extensions['review_jobs'].jobs[key].source.exists()
