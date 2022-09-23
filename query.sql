select top 2000 "telemetry.time", "deviceId", "telemetry.co2"
from [dbo].[AmbiNode] order by "IdPxKey" desc