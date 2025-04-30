# LUA API

In the `shell` tab of mmWaveStudio you can run

```lua
help command
```

and lua will print the help for the command. For convenience, here is a list of the commands available in mmWaveStudio:

## DataPathConfig

```lua
>help ar1.DataPathConfig
Int32 ar1.DataPathConfig(UInt32 intfSel, UInt32 transferFmtPkt0, UInt32 transferFmtPkt1) - DataPathConfig API Defines the used to configure the device data path
_I_ UInt32	intfSel	 - Data path interface select(0:7)+ CQ config(b8:15)
_I_ UInt32	transferFmtPkt0	 - Data output format(b0:7)+ CQ0TransSize(b8:15)+ CQ1TransSize(b16:23)+ CQ2TransSize(b24:31)
_I_ UInt32	transferFmtPkt1	 - Supress packet 1 transmission
```

## ChanNAdcConfig

```lua
>help ar1.ChanNAdcConfig

Int32 ar1.ChanNAdcConfig(UInt16 Tx0En, UInt16 Tx1En, UInt16 Tx2En, UInt16 Rx0En, UInt16 Rx1En, UInt16 Rx2En, UInt32 Rx3En, Int32 BitsVal, UInt32 FmtVal, UInt32 IQSwap) - Static device config API which defines configure both the Transmiter and Reciever channels of Radar device and also ADC data format output
_I_ UInt16 Tx0En - Tx0 channel
_I_ UInt16 Tx1En - Tx1 channel
_I_ UInt16 Tx2En - Tx2 channel
_I_ UInt16 Rx0En - Rx0 channel
_I_ UInt16 Rx1En - Rx1 channnel
_I_ UInt16 Rx2En - Rx2 channel
_I_ UInt32 Rx3En - Rx3 channel[b15:0] + (CascadePinOutCfg[b31:16] b16:ClkOutMasterDis, b17:SynOutMasterDis, b18:ClkOutSlaveEna, b19:SynOutSlaveEna, b20:IntLOMasterEna, b21:OSCClkOutMasterDis)
_I_ Int32 BitsVal - Number of ADC bits
_I_ UInt32 FmtVal - ADC output format[b15:0] + FullScaleReductionFactor[b31:16]
_I_ UInt32 IQSwap - ADC Mode[b15:0] + CascadeMode[b31:16](Single Chip: 0x0000, MultiChip Master:0x0001, MultiChip Slave:0x0002)
```
