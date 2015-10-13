# cluster example
library(parallel)

n.cores <- detectCores()

workerFunc <- function(n) { return (n^2) }

values <- 1:1000

system.time(res <- lapply(values, workerFunc))

system.time(res2 <- mclapply(values,workerFunc, mc.cores = 2))
system.time(res4 <- mclapply(values,workerFunc, mc.cores = 4))

# parLapply PSOCK
cl  <- makeCluster(4,type="PSOCK")
system.time(resCL <- parLapply(cl,values,workerFunc))
stopCluster(cl)

# other example, incomplete here...
system.time({
  clusterExport(cl, varlist = c("patient.data.mod", "bootstrap"))
  result2 <- parSapply(
    cl = cl, 
    X = 1:50000, 
    FUN = function(x) { bootstrap(data = patient.data.mod, n.boot = 1) })
})

# Toon structuur van patient.data.split
str(patient.data.split)

# Bootstrap gemiddelde per ziekenhuis-afdeling
result <- parLapply( cl = cl, 
                     X = patient.data.split, 
                     fun = bootstrap, 
                     n.boot = 1000)


# to run accross multiple machines this 
# needs doMPI, further reading...
### $mpirun -H localhost,n2n23 -n 6 R --slave -f sincMPI.R
