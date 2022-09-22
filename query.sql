select top 20 "enqueuedTime", "messageProperties.iothub-creation-time-utc" 
from [dbo].[AmbiNode] order by "IdPxKey" desc