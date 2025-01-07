from mido import MidiFile, MidiTrack, Message, MetaMessage, second2tick
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

def parse_midi_notes_sequence(file_path):
    """
    Reads a MIDI file and convert it to a sequence of individual notes (no chords, no length, no velocity)

    Args:
        file_path (str) : Path to the MIDI file

    Returns:
        list[int] : List of MIDI note numbers
    """
    midi = MidiFile(file_path)
    notes = []

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    return notes

def map_notes_to_range(notes, min_note=24, max_note=108):
    """
    Maps notes to a specified range, notes outside the range are dropped

    Args:
        notes (list[int]) : List of MIDI note numbers
        min_note (int) : MIDI note number of lowest allowed note
        max_note (int) : MIDI note number of highest allowed note
    
    Returns:
        list[int] : List of MIDI note numbers
    """
    return [note for note in notes if min_note <= note <= max_note]

def notes_to_one_hot(notes, min_note=24, max_note=108):
    """
    Converts notes from MIDI note number to one-hot vectors

    Args:
        notes (list[int]) : List of MIDI note numbers
        min_note (int) : MIDI note number of lowest allowed note
        max_note (int) : MIDI note number of highest allowed note
    
    Returns:
        numpy.ndarray: A 2D array of shape (len(notes), max_note - min_note + 1),
            where each row is a one-hot vector corresponding to a note.
    """

    num_notes = max_note - min_note + 1
    one_hot_vectors = []

    for note in notes:
        one_hot = np.zeros(num_notes)
        one_hot[note - min_note] = 1
        one_hot_vectors.append(one_hot)

    return np.array(one_hot_vectors)

def midi_to_one_hot(file_path, min_note=24, max_note=108):
    """
    Converts a MIDI file to a numpy array of one-hot vectors with the help of above helper functions

    Args:
        file_path (str) : Path to the MIDI file
        min_note (int) : MIDI note number of lowest allowed note
        max_note (int) : MIDI note number of highest allowed note
    
    Returns:
        numpy.ndarray: A 2D array of shape (len(notes), max_note - min_note + 1),
            where each row is a one-hot vector corresponding to a note.
    """

    notes = parse_midi_notes_sequence(file_path)
    mapped_notes = map_notes_to_range(notes, min_note, max_note)
    one_hot_vectors = notes_to_one_hot(mapped_notes, min_note, max_note)
    return one_hot_vectors

def midi_to_note_indices(file_path, min_note=24, max_note=108):
    midi_notes = parse_midi_notes_sequence(file_path)
    mapped_notes = map_notes_to_range(midi_notes, min_note, max_note)
    note_indices = [midi_int_to_note_index(midi_note) for midi_note in mapped_notes]
    return note_indices

def midi_int_to_note_index(midi_note):
    """
    Converts MIDI note number (between 24 and 109) to note index (between 0 and 85).

    Args:
        midi_note (int) : Note number in MIDI (between 24 and 109)
    
    Returns:
        Integer between 0 (C1) and 85 (C8) for MIDI compatibility.
    """
    return midi_note - 24

def note_index_to_midi_int(note_index):
    """
    Converts note index (between 0 and 85) to MIDI note number (between 24 and 109).

    Args:
        note_index (int) : Index of note in the one-hot vector (between 0 and 85)
    
    Returns:
        Integer between 24 (C1) and 109 (C8) for MIDI compatibility.
    """
    return 24 + note_index

def one_hot_encode(sequence, num_notes=85):
    """
    Converts a sequence of integer notes to a sequence of one-hot encoded notes.
    
    Args:
        sequence (list[int]) : List of integers with a maximum difference of num_notes. 
        num_notes (int) : Size of "vocabulary", the difference between the lowest and highest possible notes.
    
    Returns:
        numpy.ndarray: A 2D array of shape (len(sequence), num_notes),
            where each row is a one-hot vector corresponding to a note.
    """
    one_hot = np.zeros((len(sequence), num_notes), dtype=np.float32)
    for i, note in enumerate(sequence):
        one_hot[i, note] = 1.0
    return one_hot

def notes_to_midi(notes, output_file, ticks_per_beat=480):
    """
    Saves a sequence of notes to a MIDI file.

    Args:
        notes (list[int]) : Sequence of notes represented as integers between 0 and 85.
        output_file (str) : Path of output file.
        ticks_per_beat (int) : Ticks per beat in the MIDI file.
    
    Returns:
        None
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Add a simple tempo and program change (default piano instrument)
    track.append(Message('program_change', program=0, time=0))

    # Convert notes from index to MIDI note and add to the track
    for note in notes:
        midi_note = note_index_to_midi_int(note)
        track.append(Message('note_on', note=midi_note, velocity=64, time=0))
        track.append(Message('note_off', note=midi_note, velocity=64, time=480))

    # Save the MIDI file
    midi.save(output_file)
    print(f"MIDI file saved to {output_file}")

def parse_note_onoff_events(mid, tick_resolution):
    """
    Collects note on/off events from a midi object and stores them with their timing rounded to tick_resolution.

    Args:
        mid (MidiFile) : A MIDI file parsed with Mido library
        tick_resolution (int) : The resolution of sampling, so MIDI timings will be rounded based on this value (e.g. 5)
    
    Returns:
        list[(int, str, int)]: A list containing tuples of (rounded_time, note, velocity)
            for each note event in the MIDI. 
    """
    note_events = []
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if (msg.type == 'note_on' or msg.type == 'note_off') and 24 <= msg.note <= 108:
                rounded_time = round(current_time / tick_resolution) * tick_resolution
                if msg.type == 'note_on':
                    note_events.append((rounded_time, msg.note - 24, msg.velocity))
                else:
                    note_events.append((rounded_time, msg.note - 24, 0))
    
    # Sort events by time
    note_events.sort(key=lambda x: x[0])

    return note_events

def generate_vectors_from_note_events(note_events, tick_resolution):
    """
    Converts a list of note events (with their timings rounded to tick resolution) to a list of vectors,
    where each vector represents the state of every note (on/off = 1/0) in the given tick.

    Args:
        note_events (list[(int, str, int)]) : A list containing tuples of (rounded_time, note, velocity)
            for each note event in the MIDI.
        tick_resolution (int) : The resolution of sampling
    
    Returns:
        list[list[int]]: A list containing lists that contain note states of ticks of a MIDI sequence.
    """

    vectors = []
    active_notes = set()
    current_tick = 0

    for rounded_time, note, velocity in note_events:
        # Fill in vectors up to the current event's time
        while current_tick < rounded_time:
            vector = np.zeros(85, dtype=int)
            vector[list(active_notes)] = 1
            vectors.append(vector)
            current_tick += tick_resolution
        
        # Update the set of active notes (in the dataset,
        # instead of note_off events, the authors used
        # note_on events with 0 velocity)
        if velocity > 0: 
            active_notes.add(note)
        else:
            active_notes.discard(note)

    # Handle remaining time after the last event
    while current_tick <= max(e[0] for e in note_events):
        vector = np.zeros(85, dtype=int)
        vector[list(active_notes)] = 1
        vectors.append(vector)
        current_tick += tick_resolution
    
    return vectors

def midi_to_multiclass_vectors(midi_file, tick_resolution=5):
    """
    Converts a MIDI file into fixed-interval vector representation based on tick resolution.
    
    Args:
        midi_file (str): Path to the MIDI file.
        tick_resolution (int): The resolution of sampling for rounding (e.g. 5).
    
    Returns:
      list[list[int]]: A list containing lists that contain note states of ticks of a MIDI sequence.
    """
    # Read the MIDI file
    mid = MidiFile(midi_file)

    # Collect all note events with their timing
    note_events = parse_note_onoff_events(mid, tick_resolution)

    # Generate vectors for every tick based on the note events
    vectors = generate_vectors_from_note_events(note_events, tick_resolution)

    return vectors[:-1] # Cut off end to lose last note_off event


def multiclass_vectors_to_midi(vectors, output_file, tick_resolution=5):
    """
    Converts a sequence of vectors back into a MIDI file.
    
    Args:
        vectors (list[list[int]]): A 2D list where each row is a 128-length vector containing note states of ticks.
        output_file (str): Path to save the output MIDI file.
        tick_resolution (int): The resolution of sampling for rounding (e.g. 5).
    
    Returns:
        None
    """
    # Create a new MIDI file and add a track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    active_notes = set()
    last_tick = 0

    for tick_idx, vector in enumerate(vectors):
        current_tick = tick_idx * tick_resolution
        active_notes_this_tick = {note for note, value in enumerate(vector) if value == 1}

        # Add note-on events
        note_on_events = active_notes_this_tick - active_notes
        for note in note_on_events:
            delta_time = current_tick - last_tick
            track.append(Message('note_on', note=note+24, velocity=64, time=delta_time))
            last_tick = current_tick

        # Add note-off events (= note_on events with 0 velocity)
        note_off_events = active_notes - active_notes_this_tick
        for note in note_off_events:
            delta_time = current_tick - last_tick
            track.append(Message('note_on', note=note+24, velocity=0, time=delta_time))
            last_tick = current_tick

        # Update active notes
        active_notes = active_notes_this_tick
    
    total_ticks = len(vectors) * tick_resolution
    
    for note in active_notes:
        delta_time = total_ticks - last_tick
        last_tick += delta_time
        track.append(Message('note_on', note=note+24, velocity=0, time=delta_time))

    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved to {output_file}")