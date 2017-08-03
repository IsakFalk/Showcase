import model_a
import model_b
import model_c
import model_d

from part1lib import optimize_hyperparams

### Find hyperparameters

# model a

print "Run model a\n"

opt_a = optimize_hyperparams(model_a.run_model_a)

# model b

print "Run model b\n"

opt_b = optimize_hyperparams(model_b.run_model_b)

# model c

print "Run model c\n"

opt_c = optimize_hyperparams(model_c.run_model_c)

# model d

print "Run model d\n"

opt_d = optimize_hyperparams(model_d.run_model_d)

print "\nAll runs finished, the optimal parameters are\nmodel a: ", opt_a, "\nmodel b: ", opt_b, "\nmodel c: ", opt_c, "\nmodel d", opt_d
