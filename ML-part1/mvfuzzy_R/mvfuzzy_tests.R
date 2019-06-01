library(readr)
source('mvfuzzy.R')

options("scipen"=999)

main <- function() {
  mfeat_fac = read_table("mfeat/mfeat-fac", col_names=F,
                         col_types=cols(.default = col_double()))
  mfeat_fou = read_table("mfeat/mfeat-fou", col_names=F,
                         col_types=cols(.default = col_double()))
  mfeat_kar = read_table("mfeat/mfeat-kar", col_names=F,
                         col_types=cols(.default = col_double()))

  # normalize
  norm_fac = scale(mfeat_fac)
  norm_fou = scale(mfeat_fou)
  norm_kar = scale(mfeat_kar)

  # compute dissimilarity matrices
  # já chequei que os dados estão IGUAIS
  D = array(0.0, dim=c(2000, 2000, 3))
  D[,, 1] = as.matrix(dist(norm_fac))
  D[,, 2] = as.matrix(dist(norm_fou))
  D[,, 3] = as.matrix(dist(norm_kar))

  for(i in seq(1)) {
    run_tests(D, 10, 1.6, 150, 10^-10)
  }
}

run_tests <- function(D, K, m, T, err) {
  result <<- mv.fuzzy(D, K, m, T, err)
}


main()
