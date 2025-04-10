# Stress Test for Linux

To build the binary:
go build

To view help:
./stress -h

## Example

./stress -cpu -memory -l /home -block 4K,512K -size 1G -mode both -duration 10m

This command will run a stress test for 10 minutes, applying CPU, memory, and disk stress on `/home`, using 4K and 512K block sizes to perform both sequential and random writes of 1GB files.

## Stress CPU only

./stress -cpu -duration 10m

## Stress memory only

./stress -memory -duration 10m

## Stress disk only (sequential, random, or both)

./stress -l /home -size 1G -block 4K -mode sequential

- `-mode` can be `sequential`, `random`, or `both`.

## Additional Notes

- `config.json` is used for debug configuration. If `"debug": true`, debug information will be shown.
- Stress test logs are saved to `stress.log`.
