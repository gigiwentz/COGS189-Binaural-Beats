"""
EEG Music Cognition Study
=========================
Within-subjects, counterbalanced design comparing:
  S = Silence
  M = Study music (beta binaural beats / brain.fm-style)
  C = Control music (non-entrainment instrumental)

Structure:
  - 2-minute EEG baseline (fixation cross, no task)
  - 10 blocks x 3 trials x ~10s = ~5 minutes of task
  - Conditions (S, M, C) assigned to blocks, counterbalanced across subjects
  - PsychoPy handles audio playback, photosensor dot marks trial boundaries
  - Participant plays Connections game in external browser during task blocks

NOTE: Place your audio files at the paths defined in AUDIO_FILES below.
      The beta beats track was inspired by "Power Focus - 14Hz Beta Waves"
      (first result for "beta binaural beats music"). Timing/track may be
      swapped — see comments marked # AUDIO_SWAP.
"""

from psychopy import visual, core, sound, event
from psychopy.hardware import keyboard
import numpy as np
from scipy import signal
import random, os, pickle, time
import mne

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
cyton_in        = True
width           = 1536
height          = 864
refresh_rate    = 60.02
subject         = 1       # Change per participant
session         = 1

# Counterbalancing: Latin square row index (0–5) assigned per subject.
# With 3 conditions there are 6 possible orderings (rows of the Latin square).
# Assign subject 1→row 0, subject 2→row 1, ... subject 7→row 0, etc.
LATIN_SQUARE = [
    ['S', 'M', 'C'],
    ['S', 'C', 'M'],
    ['M', 'S', 'C'],
    ['M', 'C', 'S'],
    ['C', 'S', 'M'],
    ['C', 'M', 'S'],
]
counterbalance_row = (subject - 1) % 6  # automatically cycles through rows

# Blocks per condition — adjust as needed (must sum to n_blocks)
n_blocks        = 10
# Default: 3 S, 3 M, 4 C — change freely, must sum to n_blocks
blocks_per_condition = {'S': 3, 'M': 3, 'C': 4}
assert sum(blocks_per_condition.values()) == n_blocks, \
    "blocks_per_condition must sum to n_blocks"

# Trial timing
n_trials_per_block  = 3
trial_duration      = 10.0   # seconds
iti_duration        = 0.5    # inter-trial interval (s)
baseline_duration   = 120.0  # 2 minutes

# Audio files — swap paths or filenames here as needed
# AUDIO_SWAP: replace paths with your actual audio files
AUDIO_FILES = {
    'M': 'audio/beta_binaural_beats.mp3',   # AUDIO_SWAP: study music (14Hz beta beats)
    'C': 'audio/control_instrumental.mp3',  # AUDIO_SWAP: control (non-entrainment)
    'S': None,                               # Silence — no file needed
}

# Save paths
save_dir = f'data/music_eeg/sub-{subject:02d}/ses-{session:02d}/'
os.makedirs(save_dir, exist_ok=True)
save_file_eeg        = save_dir + 'eeg_raw.npy'
save_file_aux        = save_dir + 'aux_raw.npy'
save_file_eeg_trials = save_dir + 'eeg_trials.npy'
save_file_aux_trials = save_dir + 'aux_trials.npy'
save_file_metadata   = save_dir + 'metadata.npy'
save_file_events     = save_dir + 'solve_events.npy'

# ─────────────────────────────────────────────
# OPENBCI CYTON SETUP
# ─────────────────────────────────────────────
if cyton_in:
    import glob, sys, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial
    from threading import Thread, Event
    from queue import Queue

    sampling_rate    = 250
    CYTON_BOARD_ID   = 0
    BAUD_RATE        = 115200
    ANALOGUE_MODE    = '/2'

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

    def get_data(queue_in):
        while not stop_event.is_set():
            data_in      = board.get_board_data()
            timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg_in       = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux_in       = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(timestamp_in) > 0:
                print(f'[EEG thread] eeg:{eeg_in.shape} aux:{aux_in.shape} ts:{timestamp_in.shape}')
                queue_in.put((eeg_in, aux_in, timestamp_in))
            time.sleep(0.1)

    cyton_thread = Thread(target=get_data, args=(queue_in,))
    cyton_thread.daemon = True
    cyton_thread.start()

# ─────────────────────────────────────────────
# BUILD BLOCK SEQUENCE (counterbalanced)
# ─────────────────────────────────────────────
def build_block_sequence(blocks_per_condition, counterbalance_row, seed=None):
    """
    Distributes condition labels across n_blocks.
    Uses the Latin square row to determine the ORDER in which conditions
    first appear, then shuffles within that constraint so no two consecutive
    blocks share the same condition where possible.
    """
    condition_order = LATIN_SQUARE[counterbalance_row]  # e.g. ['M', 'S', 'C']
    block_list = []
    for cond in condition_order:
        block_list.extend([cond] * blocks_per_condition[cond])

    # Shuffle with seed = subject so it's reproducible but unique per subject
    rng = random.Random(seed if seed is not None else subject)
    rng.shuffle(block_list)

    # Attempt to avoid consecutive same-condition runs (best-effort)
    for _ in range(1000):
        has_repeat = any(block_list[i] == block_list[i+1] for i in range(len(block_list)-1))
        if not has_repeat:
            break
        rng.shuffle(block_list)

    return block_list

block_sequence = build_block_sequence(blocks_per_condition, counterbalance_row, seed=subject)
print(f'\n[Design] Subject {subject} | Counterbalance row {counterbalance_row}')
print(f'[Design] Block sequence: {block_sequence}\n')

# ─────────────────────────────────────────────
# PSYCHOPY WINDOW & STIMULI
# ─────────────────────────────────────────────
kb     = keyboard.Keyboard()
window = visual.Window(
    size=[width, height],
    checkTiming=True,
    allowGUI=False,
    fullscr=True,
    useRetina=False,
)
aspect_ratio = width / height

fixation    = visual.TextStim(window, text='+', height=0.1, color='white', units='norm')
status_text = visual.TextStim(window, text='', pos=(0, 0), height=0.06, color='white',
                               units='norm', wrapWidth=1.8)

def photosensor_dot(window, on=False, size=2/8*0.7):
    """Returns a Rect positioned bottom-right; white=on, black=off."""
    ratio = window.size[0] / window.size[1]
    rect  = visual.Rect(win=window, units='norm',
                        width=size, height=size * ratio,
                        fillColor='white' if on else 'black',
                        lineWidth=0,
                        pos=[1 - size/2, -1 + size * ratio / 2])
    return rect

dot_on  = photosensor_dot(window, on=True)
dot_off = photosensor_dot(window, on=False)

# Audio — load only files that exist
audio_players = {}
for cond, path in AUDIO_FILES.items():
    if path and os.path.exists(path):
        audio_players[cond] = sound.Sound(path, loops=-1)  # loop continuously
    elif path:
        print(f'[WARNING] Audio file not found for condition {cond}: {path}')

# ─────────────────────────────────────────────
# EEG DATA BUFFERS
# ─────────────────────────────────────────────
eeg       = np.zeros((8, 0))
aux       = np.zeros((3, 0))
timestamp = np.zeros((0,))
eeg_trials = []
aux_trials = []
trial_metadata = []  # (block_i, trial_i, condition, trial_start_sample)
solve_events   = []  # {'block', 'trial', 'condition', 'time_since_trial_start', 'eeg_sample'}

def drain_queue():
    """Pull all pending data from the Cyton thread into buffers."""
    global eeg, aux, timestamp
    while not queue_in.empty():
        eeg_in, aux_in, ts_in = queue_in.get()
        eeg       = np.concatenate((eeg, eeg_in), axis=1)
        aux       = np.concatenate((aux, aux_in), axis=1)
        timestamp = np.concatenate((timestamp, ts_in))

def get_trial_boundaries():
    photo_trigger = (aux[1] > 20).astype(int)
    starts = np.where(np.diff(photo_trigger) ==  1)[0]
    ends   = np.where(np.diff(photo_trigger) == -1)[0]
    return starts, ends

def save_all():
    np.save(save_file_eeg,        eeg)
    np.save(save_file_aux,        aux)
    np.save(save_file_eeg_trials, np.array(eeg_trials, dtype=object))
    np.save(save_file_aux_trials, np.array(aux_trials, dtype=object))
    np.save(save_file_metadata,   np.array(trial_metadata, dtype=object))
    np.save(save_file_events,     np.array(solve_events,   dtype=object))
    print(f'[Saved] {save_dir}')

def quit_clean():
    if cyton_in:
        stop_event.set()
        board.stop_stream()
        board.release_session()
    save_all()
    window.close()
    core.quit()

def check_escape():
    keys = kb.getKeys()
    if 'escape' in keys:
        quit_clean()

def stop_all_audio():
    for player in audio_players.values():
        try:
            player.stop()
        except Exception:
            pass

def extract_and_store_trial(trial_index, condition, skip_count=0):
    """Filter, epoch, baseline-correct and store the most recent trial."""
    baseline_samp  = int(0.2 * sampling_rate)
    trial_samp     = int(trial_duration * sampling_rate) + baseline_samp
    starts, ends   = get_trial_boundaries()
    idx            = trial_index + skip_count
    if idx >= len(starts):
        print(f'[WARNING] Trial boundary not found for trial {trial_index}')
        return
    t_start = starts[idx] - baseline_samp
    filtered = mne.filter.filter_data(eeg, sfreq=sampling_rate,
                                       l_freq=2, h_freq=40, verbose=False)
    t_eeg = np.copy(filtered[:, t_start:t_start + trial_samp])
    t_aux = np.copy(aux[:,     t_start:t_start + trial_samp])
    baseline_avg = np.mean(t_eeg[:, :baseline_samp], axis=1, keepdims=True)
    t_eeg -= baseline_avg
    eeg_trials.append(t_eeg)
    aux_trials.append(t_aux)
    print(f'[Trial] idx={trial_index} cond={condition} shape={t_eeg.shape}')

# ─────────────────────────────────────────────
# PHASE 1: BASELINE (2 minutes, fixation cross)
# ─────────────────────────────────────────────
print('\n[Phase] Baseline — 2 minutes fixation cross')
status_text.text = 'Baseline recording\nPlease fixate on the cross and remain still.'
status_text.draw()
fixation.draw()
dot_off.draw()
window.flip()
core.wait(2.0)  # brief pause before baseline starts

baseline_clock = core.Clock()
while baseline_clock.getTime() < baseline_duration:
    check_escape()
    elapsed = baseline_clock.getTime()
    remaining = baseline_duration - elapsed
    status_text.text = f'Baseline: {int(remaining)}s remaining\nPlease relax and fixate.'
    status_text.draw()
    fixation.draw()
    dot_off.draw()
    window.flip()
    if cyton_in:
        drain_queue()

print('[Phase] Baseline complete.')

# ─────────────────────────────────────────────
# PHASE 2: TASK BLOCKS
# ─────────────────────────────────────────────
total_trial_index = 0  # global counter for photosensor boundary indexing

for i_block, condition in enumerate(block_sequence):

    # ── Condition transition screen ──────────────────────────────────────
    cond_labels = {'S': 'SILENCE', 'M': 'STUDY MUSIC', 'C': 'CONTROL MUSIC'}
    print(f'\n[Block {i_block+1}/{n_blocks}] Condition: {cond_labels[condition]}')

    status_text.text = (f'Block {i_block+1} of {n_blocks}\n'
                        f'Condition: {cond_labels[condition]}\n\n'
                        f'Open the Connections game in your browser.\n'
                        f'Press SPACE when ready.')
    status_text.draw()
    dot_off.draw()
    window.flip()

    # Wait for spacebar
    kb.clearEvents()
    while True:
        check_escape()
        keys = kb.getKeys(['space'])
        if keys:
            break
        status_text.draw()
        dot_off.draw()
        window.flip()

    # ── Start audio for this condition ───────────────────────────────────
    stop_all_audio()
    if condition in audio_players:
        audio_players[condition].play()
        print(f'[Audio] Playing: {AUDIO_FILES[condition]}')
    else:
        print('[Audio] Silence — no audio playing.')

    # ── Run trials within block ──────────────────────────────────────────
    for i_trial in range(n_trials_per_block):

        print(f'  [Trial {i_trial+1}/{n_trials_per_block}] block={i_block+1} cond={condition}')

        # Brief ITI — dot off
        dot_off.draw()
        window.flip()
        core.wait(iti_duration)

        # Trial onset — dot ON (photosensor trigger)
        trial_clock = core.Clock()
        while trial_clock.getTime() < trial_duration:
            check_escape()
            # SPACE = participant solved the puzzle
            keys = kb.getKeys(['space'])
            if keys:
                solve_time = trial_clock.getTime()
                solve_sample = eeg.shape[1]  # current EEG sample index
                solve_events.append({
                    'block':                i_block,
                    'trial':                i_trial,
                    'condition':            condition,
                    'time_since_trial_start': solve_time,
                    'eeg_sample':           solve_sample,
                })
                print(f'  [SOLVE] block={i_block+1} trial={i_trial+1} '
                      f'cond={condition} t={solve_time:.2f}s')
            dot_on.draw()
            window.flip()

        # Trial offset — dot off
        dot_off.draw()
        window.flip()

        # ── Collect EEG for this trial ───────────────────────────────────
        if cyton_in:
            # Wait until we have at least this trial's boundary
            wait_clock = core.Clock()
            while True:
                drain_queue()
                _, ends = get_trial_boundaries()
                if len(ends) > total_trial_index:
                    break
                if wait_clock.getTime() > 5.0:
                    print('[WARNING] Timeout waiting for trial boundary.')
                    break
                time.sleep(0.05)

            extract_and_store_trial(total_trial_index, condition)
            trial_metadata.append({
                'block':     i_block,
                'trial':     i_trial,
                'condition': condition,
                'global_trial_index': total_trial_index,
            })

        total_trial_index += 1

    # ── End of block — stop audio, brief rest ────────────────────────────
    stop_all_audio()
    print(f'[Block {i_block+1}] Complete.')

    if i_block < n_blocks - 1:
        status_text.text = 'Block complete. Take a short break.\n\nPress SPACE to continue.'
        status_text.draw()
        dot_off.draw()
        window.flip()
        kb.clearEvents()
        while True:
            check_escape()
            keys = kb.getKeys(['space'])
            if keys:
                break
            status_text.draw()
            dot_off.draw()
            window.flip()

# ─────────────────────────────────────────────
# PHASE 3: END SCREEN
# ─────────────────────────────────────────────
status_text.text = 'Study complete. Thank you!\n\nPlease wait for the experimenter.'
status_text.draw()
dot_off.draw()
window.flip()
core.wait(3.0)

# ─────────────────────────────────────────────
# SAVE & CLEAN UP
# ─────────────────────────────────────────────
if cyton_in:
    drain_queue()
save_all()
quit_clean()
