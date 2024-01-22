# load("R_packages.rda")
installedpackages = c('argparse', 'sirus')
for (count in 1:length(installedpackages)) {
    install.packages(installedpackages[count],repos="http://cran.us.r-project.org")
}
