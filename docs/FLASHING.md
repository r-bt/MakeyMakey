When flashing DEMO .bin onto the board remember to put jumpers on SOP2 and SOP0

- Source: MMWAVE SDK User Guide see functional (demonstration mode)

## MSS Power Up async event was not received

When using a .LUA script in mmWave you may get the following error

```
[10:35:50]  Status: Failed, Error Type: RESP TIMEOUT
[10:35:55]  MSS Power Up async event was not received!
```

Generally happens if DEMO .bin was flashed previosuly, you first have to use UNIFLASH
to format the device