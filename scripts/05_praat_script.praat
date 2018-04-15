form Script arguments
	sentence Wav_path
	sentence interp_F0_path
endform


wav_file = Read from file: wav_path$
selectObject: wav_file

To Pitch (ac)... 0.005 75.0 15 yes 0.03 0.5 0.01 0.4 0.14 600.0
Down to PitchTier

start_time = Get start time
start_f0 = Get value at time... start_time
Add point... start_time start_f0

end_time = Get end time
end_f0 = Get value at time... end_time 
Add point... end_time end_f0 

To Pitch... 0.005 60.0 600.0

Smooth... 10.0
Interpolate

Down to PitchTier

end_time = Get end time
end_f0 = Get value at time... end_time 
Add point... end_time end_f0 

Shift times by... -0.0025

Save as headerless spreadsheet file: interp_F0_path$
