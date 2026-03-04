"""
EEG Music Cognition Study
=========================
Within-subjects, counterbalanced design comparing:
  S = Silence
  M = Study music (beta binaural beats / brain.fm-style)
  C = Control music (non-entrainment instrumental)

Per-condition structure (~7 min each, ~21 min total):
  ┌──────────────────────────────────────────────────────────────┐
  │  2 min BASELINE  │ 10s BUF │ 30s PRE │  5 min TASK  │ 30s POST │
  │  (eyes open,     │ (trans- │ (eyes   │ (Connections │ (eyes    │
  │   fixation)      │  ition) │  open)  │  ±15s epochs)│  open)   │
  └──────────────────────────────────────────────────────────────┘
  BASELINE  : fixation cross, eyes open, no task, no audio
  BUFFER    : 10s transition — audio starts here, not saved
  PRE-REC   : 30s eyes-open recording before task (no puzzle)
  TASK      : participant plays Connections in browser;
              SPACE key = puzzle solved → epoch ±15s saved
  POST-REC  : 30s eyes-open recording after task (no puzzle)

Audio:
  PsychoPy auto-plays looping audio for M and C conditions.
  AUDIO_SWAP comments mark where to change audio file paths.

Counterbalancing:
  Latin square (6 rows) assigned automatically by subject number.
  Condition order is unique and reproducible per subject.
"""

from psychopy import visual, core, sound
from psychopy.hardware import keyboard
import numpy as np
import random, os, time
import mne

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
cyton_in  = True
width     = 1536
height    = 864
subject   = 1    # ← change per participant
session   = 1

# Timing (seconds)
# Per condition = 2 min baseline + 10s buffer + 5 min task = 7 min 10s
#
# Audio plays for 5 min 10s = buffer + full task block
#
# 5 min task block — participant plays Connections the entire time:
#   0:00–0:30  → PRE-REC  (first 30s, EEG saved, photosensor ON)
#   0:30–4:30  → MAIN     (240s, EEG saved, photosensor ON)
#   4:30–5:00  → POST-REC (last 30s, EEG saved, photosensor ON)
#   total task = 300s = 5 min exactly
baseline_duration   = 120.0  # 2 min baseline (fixation, no task, no audio)
buffer_duration     =  10.0  # 10s buffer — audio starts, task not yet open, not saved
total_task_duration = 300.0  # 5 min task block (participant plays Connections throughout)
pre_rec_duration    =  30.0  # first 30s of task — EEG saved
post_rec_duration   =  30.0  # last 30s of task  — EEG saved
# derived
main_task_duration  = total_task_duration - pre_rec_duration - post_rec_duration  # 240s
solve_epoch_window  =  15.0  # ±15s around each solve keypress

# Counterbalancing — Latin square, 3 conditions × 6 orderings
COND_LABELS  = {'S': 'SILENCE', 'M': 'STUDY MUSIC', 'C': 'CONTROL MUSIC'}
LATIN_SQUARE = [
    ['S', 'M', 'C'],
    ['S', 'C', 'M'],
    ['M', 'S', 'C'],
    ['M', 'C', 'S'],
    ['C', 'S', 'M'],
    ['C', 'M', 'S'],
]
condition_order = LATIN_SQUARE[(subject - 1) % 6]

# Audio files
# AUDIO_SWAP: replace these paths with your actual audio files
AUDIO_FILES = {
    'M': 'audio/beta_binaural_beats.mp3',   # AUDIO_SWAP: 14Hz beta beats study music
    'C': 'audio/control_instrumental.mp3',  # AUDIO_SWAP: non-entrainment control music
    'S': None,                               # Silence — no file needed
}

# Save paths
save_dir = f'data/music_eeg/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)

# ─────────────────────────────────────────────
# OPENBCI CYTON SETUP
# ─────────────────────────────────────────────
if cyton_in:
    import glob, sys, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    sampling_rate  = 250
    CYTON_BOARD_ID = 0
    BAUD_RATE      = 115200
    ANALOGUE_MODE  = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS')
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if 'OpenBCI' in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError('Cannot find OpenBCI port.')

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)

    stop_event = Event()
    queue_in   = Queue()

    def _get_data(q):
        while not stop_event.is_set():
            data_in = board.get_board_data()
            ts_in   = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in  = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in  = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(ts_in) > 0:
                q.put((eeg_in, aux_in, ts_in))
            time.sleep(0.1)

    Thread(target=_get_data, args=(queue_in,), daemon=True).start()

# ─────────────────────────────────────────────
# PSYCHOPY WINDOW & STIMULI
# ─────────────────────────────────────────────
kb = keyboard.Keyboard()
window = visual.Window(
    size=[width, height],
    checkTiming=True,
    allowGUI=False,
    fullscr=True,
    useRetina=False,
)

fixation   = visual.TextStim(window, text='+', height=0.12, color='white', units='norm')
msg        = visual.TextStim(window, text='', pos=(0, 0.1), height=0.06,
                              color='white', units='norm', wrapWidth=1.8)
timer_text = visual.TextStim(window, text='', pos=(0, -0.85), height=0.05,
                              color='gray', units='norm')

_ds = 2/8 * 0.7
_dr = width / height
dot_on  = visual.Rect(win=window, units='norm', width=_ds, height=_ds*_dr,
                      fillColor='white', lineWidth=0,
                      pos=[1 - _ds/2, -1 + _ds*_dr/2])
dot_off = visual.Rect(win=window, units='norm', width=_ds, height=_ds*_dr,
                      fillColor='black', lineWidth=0,
                      pos=[1 - _ds/2, -1 + _ds*_dr/2])

# Load audio (only files that exist)
audio_players = {}
for cond, path in AUDIO_FILES.items():
    if path and os.path.exists(path):
        audio_players[cond] = sound.Sound(path, loops=-1)
    elif path:
        print(f'[WARNING] Audio not found for condition {cond}: {path}')

# ─────────────────────────────────────────────
# EEG DATA BUFFERS & STORAGE
# ─────────────────────────────────────────────
eeg       = np.zeros((8, 0))
aux       = np.zeros((3, 0))
timestamp = np.zeros((0,))

pre_rec_epochs  = {}  # condition → filtered EEG array (30s)
post_rec_epochs = {}  # condition → filtered EEG array (30s)
solve_events    = []  # one dict per solve keypress
segment_log     = []  # phase-level log with sample indices

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def drain_queue():
    global eeg, aux, timestamp
    while not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg       = np.concatenate((eeg, eeg_in), axis=1)
        aux       = np.concatenate((aux, aux_in), axis=1)
        timestamp = np.concatenate((timestamp, ts_in))

def stop_all_audio():
    for player in audio_players.values():
        try:
            player.stop()
        except Exception:
            pass

def save_all():
    np.save(save_dir + 'eeg_raw.npy',         eeg)
    np.save(save_dir + 'aux_raw.npy',          aux)
    np.save(save_dir + 'timestamp_raw.npy',    timestamp)
    np.save(save_dir + 'pre_rec_epochs.npy',   np.array(list(pre_rec_epochs.items()),  dtype=object))
    np.save(save_dir + 'post_rec_epochs.npy',  np.array(list(post_rec_epochs.items()), dtype=object))
    np.save(save_dir + 'solve_events.npy',     np.array(solve_events, dtype=object))
    np.save(save_dir + 'segment_log.npy',      np.array(segment_log,  dtype=object))
    print(f'[Saved] → {save_dir}')

def quit_clean():
    stop_all_audio()
    if cyton_in:
        stop_event.set()
        board.stop_stream()
        board.release_session()
    save_all()
    window.close()
    core.quit()

def check_escape():
    if 'escape' in kb.getKeys(['escape']):
        print('[Escape] Saving and quitting...')
        quit_clean()

def wait_for_space(prompt):
    """Show prompt on screen and block until SPACE is pressed."""
    kb.clearEvents()
    while True:
        check_escape()
        msg.text = prompt
        msg.draw()
        dot_off.draw()
        window.flip()
        if kb.getKeys(['space']):
            break

def run_fixation_phase(duration, phase_label):
    """
    Show fixation cross + countdown for `duration` seconds.
    Photosensor dot stays OFF.
    Returns (start_sample, end_sample).
    """
    if cyton_in:
        drain_queue()
    start_sample = eeg.shape[1]
    clock = core.Clock()
    print(f'  [{phase_label}] {duration}s — samples from {start_sample}')
    while clock.getTime() < duration:
        check_escape()
        remaining = duration - clock.getTime()
        fixation.draw()
        timer_text.text = f'{phase_label}   {int(remaining)}s remaining'
        timer_text.draw()
        dot_off.draw()
        window.flip()
        if cyton_in:
            drain_queue()
    if cyton_in:
        drain_queue()
    end_sample = eeg.shape[1]
    print(f'  [{phase_label}] Done — end sample {end_sample}')
    return start_sample, end_sample

def filter_epoch(start_sample, end_sample):
    """Return a bandpass-filtered copy of the EEG between two sample indices."""
    chunk    = np.copy(eeg[:, start_sample:end_sample])
    filtered = mne.filter.filter_data(chunk, sfreq=sampling_rate,
                                       l_freq=2, h_freq=40, verbose=False)
    return filtered

# ─────────────────────────────────────────────
# INTRO SCREEN
# ─────────────────────────────────────────────
print(f'\n[Design] Subject {subject} | Condition order: {condition_order}')
wait_for_space(
    f'EEG Music Study\n\n'
    f'Subject {subject}   Session {session}\n'
    f'Condition order: {" → ".join(COND_LABELS[c] for c in condition_order)}\n\n'
    f'Press SPACE to begin.'
)

# ─────────────────────────────────────────────
# MAIN LOOP — one iteration per condition (~7 min each)
# ─────────────────────────────────────────────
for i_cond, condition in enumerate(condition_order):
    label = COND_LABELS[condition]
    print(f'\n{"="*55}')
    print(f'[Condition {i_cond+1}/3]  {label}')
    print(f'{"="*55}')

    # ── Condition start ──────────────────────────────────────────────────
    wait_for_space(
        f'Condition {i_cond+1} of 3:   {label}\n\n'
        f'You will see a fixation cross for 2 minutes.\n'
        f'Please relax and keep your eyes on the cross.\n\n'
        f'Press SPACE to begin.'
    )

    # ── PHASE 1: 2-min BASELINE (no audio) ───────────────────────────────
    b_start, b_end = run_fixation_phase(baseline_duration, 'BASELINE')
    segment_log.append({'phase': 'baseline', 'condition': condition,
                        'start': b_start, 'end': b_end})

    # ── Start audio ───────────────────────────────────────────────────────
    stop_all_audio()
    if condition in audio_players:
        audio_players[condition].play()
        print(f'  [Audio] Playing: {AUDIO_FILES[condition]}')
    else:
        print('  [Audio] Silence — no playback')

    # ── PHASES 2–5: BUFFER then 5-min TASK BLOCK ─────────────────────────
    # Participant opens Connections before pressing SPACE.
    # Audio is already playing. Buffer runs first (not saved), then
    # the full 5-min task block begins immediately after.

    wait_for_space(
        f'Condition: {label}\n\n'
        f'Open the Connections game in your browser now.\n\n'
        f'Press SPACE each time you complete a puzzle category.\n\n'
        f'[Experimenter: confirm participant is on the Connections page]\n'
        f'Press SPACE to start.'
    )

    # ── BUFFER: 10s (audio playing, task open, not saved) ────────────────
    print('  [BUFFER] 10s — audio playing, not saved')
    buf_clock = core.Clock()
    while buf_clock.getTime() < buffer_duration:
        check_escape()
        remaining = buffer_duration - buf_clock.getTime()
        timer_text.text = f'Starting in {int(remaining)+1}s...'
        timer_text.draw()
        dot_off.draw()
        window.flip()
        if cyton_in:
            drain_queue()

    # ── 5-MIN TASK BLOCK ──────────────────────────────────────────────────
    if cyton_in:
        drain_queue()

    block_clock = core.Clock()
    solve_count = 0

    pre_start_sample  = None
    post_start_sample = None

    print(f'  [TASK BLOCK] 5 min starting — {label}')

    while block_clock.getTime() < total_task_duration:
        check_escape()
        t         = block_clock.getTime()
        remaining = total_task_duration - t

        # Determine sub-phase
        if t < pre_rec_duration:
            sub_phase = 'PRE-REC'
            if pre_start_sample is None:
                if cyton_in: drain_queue()
                pre_start_sample = eeg.shape[1]
                print(f'  [PRE-REC] Started — sample {pre_start_sample}')
        elif t < total_task_duration - post_rec_duration:
            sub_phase = 'TASK'
        else:
            sub_phase = 'POST-REC'
            if post_start_sample is None:
                if cyton_in: drain_queue()
                post_start_sample = eeg.shape[1]
                print(f'  [POST-REC] Started — sample {post_start_sample}')

        # Solve keypress — logged across all sub-phases
        keys = kb.getKeys(['space'])
        if keys:
            solve_time = t
            if cyton_in:
                drain_queue()
            solve_sample = eeg.shape[1]
            solve_count += 1
            epoch_start = max(0, solve_sample - int(solve_epoch_window * sampling_rate))
            epoch_end   = solve_sample + int(solve_epoch_window * sampling_rate)
            solve_events.append({
                'condition':       condition,
                'condition_index': i_cond,
                'solve_number':    solve_count,
                'sub_phase':       sub_phase,
                'time_in_task':    solve_time,
                'eeg_sample':      solve_sample,
                'epoch_start':     epoch_start,
                'epoch_end':       epoch_end,
            })
            print(f'  [SOLVE #{solve_count}] phase={sub_phase}  '
                  f't={solve_time:.1f}s  sample={solve_sample}')

        timer_text.text = (f'{label}   {sub_phase}   {int(remaining)}s left   '
                           f'Solves: {solve_count}')
        timer_text.draw()
        dot_on.draw()
        window.flip()
        if cyton_in:
            drain_queue()

    if cyton_in:
        drain_queue()
    block_end_sample = eeg.shape[1]

    # Save pre/post epochs
    if pre_start_sample is not None:
        pre_end_sample = pre_start_sample + int(pre_rec_duration * sampling_rate)
        pre_rec_epochs[condition] = filter_epoch(pre_start_sample, pre_end_sample)
        segment_log.append({'phase': 'pre_rec', 'condition': condition,
                            'start': pre_start_sample, 'end': pre_end_sample})

    if post_start_sample is not None:
        post_end_sample = post_start_sample + int(post_rec_duration * sampling_rate)
        post_rec_epochs[condition] = filter_epoch(post_start_sample, post_end_sample)
        segment_log.append({'phase': 'post_rec', 'condition': condition,
                            'start': post_start_sample, 'end': post_end_sample})

    segment_log.append({'phase': 'full_block', 'condition': condition,
                        'start': pre_start_sample, 'end': block_end_sample,
                        'n_solves': solve_count})
    print(f'  [TASK BLOCK] Complete — {solve_count} solves')

    stop_all_audio()

    # ── Inter-condition break ─────────────────────────────────────────────
    if i_cond < len(condition_order) - 1:
        wait_for_space(
            f'Condition {i_cond+1} complete.\n\n'
            f'Take a short break — stretch, relax.\n\n'
            f'Press SPACE when ready for the next condition.'
        )

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
msg.text = 'Study complete.\nThank you!\n\nPlease wait for the experimenter.'
msg.draw()
dot_off.draw()
window.flip()
core.wait(3.0)

if cyton_in:
    drain_queue()
save_all()
quit_clean()
