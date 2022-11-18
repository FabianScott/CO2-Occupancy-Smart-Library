library(ctsmr)
 f = function(Xin, params){
  Ci = Xin[1]; rho = Xin[2]; dt = Xin[3]; n = Xin[4];
  Q = params[1];c_out = params[2];m = params[3];
  return (m*n*dt*rho + V*rho*Ci+Q*dt*C_out)/(Q*dt+V*rho)
 }
 

obs = function(C)

dX ~ f(c,r,dt,n,q,co,m) 

model = ctsm()
model$addSystem(dX)

model$addObs(formula)

