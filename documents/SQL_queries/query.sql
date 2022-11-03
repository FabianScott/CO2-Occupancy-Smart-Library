select top 2000 "telemetry.time", "telemetry.co2", "deviceId"
from [dbo].[AmbiNode] order by "IdPxKey" desc