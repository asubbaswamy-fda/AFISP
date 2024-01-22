library(sirus)
library(argparse)


parser <- ArgumentParser()
parser$add_argument("--input",dest="input_file", nargs=1, help="Input file with subgroup data")
parser$add_argument("--output",dest="output_file", nargs=1, help="Output file for sirus rules")
parser$add_argument("--depth", dest="depth", nargs=1, default=2, help="Max number of literals in rule", type="integer")
parser$add_argument("--cv", action="store_true", default=FALSE, help="Run sirus.cv to determine p0")
parser$add_argument("--rule.max", dest="max_rules", nargs=1, default=50, help="Max number of rules considered",
                   type="integer")
parser$add_argument("--p0", dest="p0", nargs=1, default=-1, help="Selection threshold",
                    type="numeric")


args <- parser$parse_args()
print(args$depth)


sirus_df <- read.csv(args$input_file)


# last column is worst-performing subset membership indicator
subset_membership <- sirus_df[, ncol(sirus_df)]
# predict subset membership from X (subgroup defining features)
sirus_X <- sirus_df[, -c(ncol(sirus_df))]

if(args$cv){
    scv <- sirus.cv(sirus_X, subset_membership, max.depth=args$depth, num.rule.max=args$max_rules)
    print("p0=")
    print(scv$p0.pred)
    sf <- sirus.fit(sirus_X, subset_membership, p0=scv$p0.pred, max.depth=args$depth, num.rule.max=args$max_rules)
} else {
    if(args$p0 == -1){
        sf <- sirus.fit(sirus_X, subset_membership, max.depth=args$depth, num.rule.max=args$max_rules)
    } else {
        sf <- sirus.fit(sirus_X, subset_membership, max.depth=args$depth, p0=args$p0)
    }
}

cat(sirus.print(sf), file=args$output_file, sep="\n")