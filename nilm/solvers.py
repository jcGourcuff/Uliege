import numpy as np
import pandas as pd
from pyomo.environ import *

SOLVER = "gurobi"

class Solver:
    """
    This class implements solvers for the LP and QP problems used througout the project.
    All of these solvers are implemented using the pyomo api and use gurobi as the core solver.
    Each problem formulation can be found in the markdown solvers.md.
    This class only contains static methods.

    :WARNING: Please be sure to have set the solver name accordingly to what's available on your machine.
              Any solver will do for the KP and RKP problems, however the label assignement problem needs a solver that handles quadratic objective functions.
    """

    def __init__(self):
        pass

    @staticmethod
    def KP(weights, profits, c):
        """
        This function solves the traditional knapsack problem.

        :param weights: Array like. The weights associated to each entity.
        :param profits: Array like. The profits associated to each entity.
        :param c: Positive real. The budget.
        :return: Tuple. An array of booleans indicating for each entity if it has been or not "put in the knapsack" and the total profit reached.
        """
        model = ConcreteModel()
        N = len(weights)
        model.ITEMS = RangeSet(0, N - 1)
        model.X = Var(model.ITEMS, domain=Boolean)

        def w_init(model, i): return weights[i]
        def p_init(model, i): return profits[i]

        model.weights = Param(model.ITEMS, initialize=w_init, within=Reals)
        model.profits = Param(model.ITEMS, initialize=p_init, within=Reals)

        model.OBJ = Objective(
            sense=maximize, expr=summation(model.profits, model.X))
        model.cons1 = Constraint(expr=summation(
            model.X, model.weights, index=model.ITEMS) <= c)

        model.OBJ.pprint()
        opt = SolverFactory(SOLVER)
        opt.solve(model)

        result = [value(model.X[k]) for k in model.ITEMS]
        z = np.sum([x * p for x, p in zip(result, profits)])

        return result, z

    @staticmethod
    def test_KP():
        """
        Function that test Solver.KP over a few examples.
        The test files were taken from https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html and can be found in data/KP_test/.

        :return: None.
        """
        root = 'data/KP_test/'
        for k in range(1, 9):
            path = root + str(k) + '/p0' + str(k) + '_'
            weights = pd.read_csv(
                path + 'w.txt', header=None).to_numpy().reshape(1, -1)[0]
            profits = pd.read_csv(
                path + 'p.txt', header=None).to_numpy().reshape(1, -1)[0]
            c = pd.read_csv(
                path + 'c.txt', header=None).to_numpy().reshape(1, -1)[0][0]

            result, _ = Solver.KP(weights, profits, c)
            print("test " + str(k))

            solution = pd.read_csv(
                path + 's.txt', header=None).to_numpy().reshape(1, -1)[0]

            if np.equal(result, solution).all():
                print("OK")
            else:
                print("Result was " + str(result) +
                      "\nsolution is " + str(solution))

    @staticmethod
    def RKP(weights, delta_w,  profits, c, gamma):
        """
        This method implements a cardinality constrained robust version of the knapsack problem.
        Plese refer to the markdown for further details.

        :param weights: Array like. The weight associated to each entity.
        :param delta_w: Array like. The maximum deviation of each weight.
        :param profits: Array like. The profit associated to each entity.
        :param c: Positive real. The budget.
        :Gamma: Float. The maximu number of weight that deviate from their nominal values.
        :return: Tuple. An array of booleans indicating for each entity if it has been or not "put in the knapsack" and the total profit reached.
        """
        weights_up = [delta_w * w for w in weights]
        model = ConcreteModel()
        N = len(weights)
        model.ITEMS = RangeSet(0, N - 1)
        model.X = Var(model.ITEMS, domain=Boolean)
        model.prob = Var(model.ITEMS, domain=PositiveReals)
        model.ro = Var(domain=PositiveReals)

        def w_init(model, i): return weights[i]
        def p_init(model, i): return profits[i]
        def w_up_init(model, i): return weights_up[i]

        model.weights = Param(
            model.ITEMS, initialize=w_init, within=PositiveReals)
        model.profits = Param(model.ITEMS, initialize=p_init, within=Reals)
        model.weights_up = Param(
            model.ITEMS, initialize=w_up_init, within=PositiveReals)
        model.gamma = Param(initialize=gamma)
        model.c = Param(initialize=c)

        model.OBJ = Objective(
            sense=maximize, expr=summation(model.profits, model.X))

        def capacityRule(model): return summation(model.X, model.weights,
                                                  index=model.ITEMS) + summation(model.prob) + model.ro * model.gamma <= model.c
        model.cons1 = Constraint(rule=capacityRule)
        def uncertaintyRule(
            model, i): return -model.X[i] * model.weights_up[i] + model.prob[i] + model.ro >= 0
        model.cons2 = Constraint(model.ITEMS, rule=uncertaintyRule)

        opt = SolverFactory(SOLVER)
        opt.solve(model)

        result = [value(model.X[k]) for k in model.ITEMS]
        z = np.sum([x * p for x, p in zip(result, profits)])

        return result, z

    @staticmethod
    def test_RKP():
        """
        Method to test the Solver.RKP. The inputs are randomly generated and fed to both Solver.KP and Solver.RKP.
        :return: Tuple. Arrays of the mean value reached by the objective functions for different values of gamma : (nominal problem, robust problem).
        """

        optimal_values = []
        nominal_values = []

        N = 200

        c = 4000.0
        delta = .01

        gammas = np.linspace(2, 116, 10)
        for g in gammas:
            opts = []
            noms = []
            for k in range(10):

                weights = list(np.random.choice(
                    np.arange(20, 30), size=N).astype(float))
                profits = list(np.random.choice(
                    np.arange(16, 78), size=N).astype(float))
                weights_up = [delta * w for w in weights]
                X, probas, ro, z = RKP(weights, weights_up, profits, c, g)
                _, z_nom = KP(weights, profits, c)
                opts.append(z)
                noms.append(z_nom)

            optimal_values.append(np.mean(opts))
            nominal_values.append(np.mean(noms))

        nominal_values = np.array(nominal_values)
        optimal_values = np.array(optimal_values)

        return nominal_values, optimal_values

    @staticmethod
    def label_assignement(N_app, N_u, N_apps, P, A_labeled, A_unlabeled, Max_W):
        """
        Solves the label assignement problem as described in the markdown.

        :param N_app: Integer. Number of different labels.
        :param N_u: Integer. Number of boxes to label.
        :param N_apps: Array of integers of size N_app. Number of box for each label.
        :P: Array of size sum(N_apps) such that p(i,j) = P(sum(N_apps[:i]) + j)
        :A_labeled: Array of size sum(N_apps). Areas of the labeld boxes. The indexation is the same as for P.
        :A_unlabeled: Array of size N_u. Areas of the unlabeld boxes.
        :Max_W: Array of size (N_u, sum(N_apps)). Maximu values for each edge linking the labeled boxes to the unlabeled ones.
        :return: Array of booleans of size (N_u, N_app). The kth entry is the one hot encoding of the label assigned to the kth box.

        .. note: Make sure to use a solver that handles quadratic objective functions.
        """

        model = ConcreteModel()

        model.N_app = Param(initialize=N_app)
        model.ITEM_I = RangeSet(0, N_app - 1)

        def N_apps_init(model, i):
            return N_apps[i]
        model.N_apps = Param(
            model.ITEM_I, initialize=N_apps_init, within=PositiveIntegers)
        model.ITEM_IJ = RangeSet(0, np.sum(N_apps) - 1)

        start_indexes = [int(np.sum(N_apps[:i])) for i in range(N_app)]

        def starting_index_init(model, i):
            return start_indexes[i]
        model.starting_index = Param(
            model.ITEM_I, initialize=starting_index_init, within=NonNegativeIntegers)

        model.ITEM_K = RangeSet(0, N_u - 1)

        model.sets = [[l for l in range(
            model.starting_index[i], model.starting_index[i] + model.N_apps[i])] for i in range(N_app)]

        def P_init(model, ij):
            return P[ij]
        model.P = Param(model.ITEM_IJ, initialize=P_init,
                        within=PercentFraction)

        def A_l_init(model, ij):
            return A_labeled[ij]
        model.A_l = Param(model.ITEM_IJ, initialize=A_l_init,
                          within=PositiveReals)

        def A_u_init(model, k):
            return A_unlabeled[k]
        model.A_u = Param(model.ITEM_K, initialize=A_u_init,
                          within=PositiveReals)

        def Mx_W_init(model, k, ij):
            return Max_W[k][ij]
        model.Mx_W = Param(model.ITEM_K, model.ITEM_IJ,
                           initialize=Mx_W_init, within=PercentFraction)

        model.W = Var(model.ITEM_K, model.ITEM_IJ, domain=PercentFraction)
        model.X = Var(model.ITEM_K, model.ITEM_I, domain=Boolean)

        def WeightsBoundRule(model, k, ij):
            return model.W[k, ij] <= model.Mx_W[k, ij]
        model.WeightsBoundCons = Constraint(
            model.ITEM_K, model.ITEM_IJ, rule=WeightsBoundRule)

        def capacityRule1(model, ij):
            return sum([model.W[k, ij] for k in model.ITEM_K]) <= 1
        model.CapacityCons1 = Constraint(model.ITEM_IJ, rule=capacityRule1)

        def capacityRule2(model, k):
            return sum(sum([model.W[k, l] * model.A_l[l] for l in model.sets[i]]) for i in model.ITEM_I) <= model.A_u[k]
        model.CapacityCons2 = Constraint(model.ITEM_K, rule=capacityRule2)

        def bound_assignRule(model, k):
            return summation(Reference(model.X[k, :]), index=model.ITEM_I) <= 1
        model.bound_assignCons = Constraint(
            model.ITEM_K, rule=bound_assignRule)

        def ReconstitutionRule(model, i):
            flows = [sum([model.W[k, l] * model.P[l]
                          for l in model.sets[i]]) for k in model.ITEM_K]
            return sum([model.X[l, i] * flows[l] for l in model.ITEM_K]) <= 1
        model.ReconstitutionCons = Constraint(
            model.ITEM_I, rule=ReconstitutionRule)

        def obj_rule(model):
            return sum([sum([sum([model.X[k, i] * model.W[k, l] * model.P[l] for l in model.sets[i]]) for k in model.ITEM_K]) for i in model.ITEM_I])
        model.OBJ = Objective(sense=maximize, rule=obj_rule)

        opt = SolverFactory(SOLVER)
        opt.solve(model)

        results = [np.argmax([value(model.X[k, i])
                              for i in model.ITEM_I]) for k in range(N_u)]
        return results

    @staticmethod
    def KP_with_constraint(weights, profits, c, negate_previous_states, new_app_bound):
        """
        This method implements a variant of the knapsack problem where there is an additionnal constraint on how many entities can be chosen relatively to previous states.

        :param weights: Array like. The weights associated to each entity.
        :param profits: Array like. The profits associated to each entity.
        :param c: Positive real. The budget.
        :param negate_previous_states: Array like. Booleans that inidcates if the entities were not previously chosen.
        :param new_app_bound: Integer. Maximum number of new chosen entities relatively to the previously chosen ones.
        :return: Tuple. An array of booleans indicating for each entity if it has been or not "put in the knapsack" and the total profit reached.
        """
        model = ConcreteModel()
        N = len(weights)
        sum_profits = np.sum(profits)
        model.ITEMS = RangeSet(0, N - 1)
        model.X = Var(model.ITEMS, domain=Boolean)

        def w_init(model, i): return weights[i]
        def p_init(model, i): return 2 * profits[i] - 1
        def ps_init(model, i): return negate_previous_states[i]

        model.weights = Param(
            model.ITEMS, initialize=w_init, within=PositiveReals)
        model.profits = Param(model.ITEMS, initialize=p_init, within=Reals)
        model.previous_states = Param(
            model.ITEMS, initialize=ps_init, within=Boolean)
        model.M = Param(initialize=new_app_bound)
        model.c = Param(initialize=c)
        model.N = Param(initialize=N)
        model.sum_profits = Param(initialize=sum_profits)

        model.OBJ = Objective(sense=maximize, expr=summation(
            model.profits, model.X) + model.N - model.sum_profits)
        model.cons1 = Constraint(expr=summation(
            model.X, model.weights, index=model.ITEMS) <= model.c)
        model.cons2 = Constraint(expr=summation(
            model.X, model.previous_states, index=model.ITEMS) <= model.M)

        opt = SolverFactory(SOLVER)
        solver_res = opt.solve(model, tee=False)

        result = [value(model.X[k]) for k in model.ITEMS]
        z = np.sum([x * w for x, w in zip(result, weights)])

        return result, z

    @staticmethod
    def label_assignement_depreciated(N_app, N_u, N_apps, P, A_labeled, A_unlabeled, Max_W, jacq_coefs):
        """
        Depreciated. No longer used in any part of the code.
        """
        model = ConcreteModel()

        model.N_app = Param(initialize=N_app)
        model.ITEM_I = RangeSet(0, N_app - 1)

        def N_apps_init(model, i):
            return N_apps[i]
        model.N_apps = Param(
            model.ITEM_I, initialize=N_apps_init, within=PositiveIntegers)
        model.ITEM_IJ = RangeSet(0, np.sum(N_apps) - 1)

        start_indexes = [int(np.sum(N_apps[:i])) for i in range(N_app)]

        def starting_index_init(model, i):
            return start_indexes[i]
        model.starting_index = Param(
            model.ITEM_I, initialize=starting_index_init, within=NonNegativeIntegers)

        model.ITEM_K = RangeSet(0, N_u - 1)

        model.sets = [[l for l in range(
            model.starting_index[i], model.starting_index[i] + model.N_apps[i])] for i in range(N_app)]

        def P_init(model, ij):
            return P[ij]
        model.P = Param(model.ITEM_IJ, initialize=P_init,
                        within=PercentFraction)

        def A_l_init(model, ij):
            return A_labeled[ij]
        model.A_l = Param(model.ITEM_IJ, initialize=A_l_init,
                          within=PositiveReals)

        def A_u_init(model, k):
            return A_unlabeled[k]
        model.A_u = Param(model.ITEM_K, initialize=A_u_init,
                          within=PositiveReals)

        def Mx_W_init(model, k, ij):
            return Max_W[k][ij]
        model.Mx_W = Param(model.ITEM_K, model.ITEM_IJ,
                           initialize=Mx_W_init, within=PercentFraction)

        def j_coeffs_init(model, k, ij):
            return jacq_coefs[k][ij]
        model.j_coeffs = Param(model.ITEM_K, model.ITEM_IJ,
                               initialize=j_coeffs_init, within=PercentFraction)

        model.W = Var(model.ITEM_K, model.ITEM_IJ, domain=PercentFraction)

        def WeightsBoundRule(model, k, ij):
            return model.W[k, ij] <= model.Mx_W[k, ij]
        model.WeightsBoundCons = Constraint(
            model.ITEM_K, model.ITEM_IJ, rule=WeightsBoundRule)

        def capacityRule1(model, ij):
            return sum([model.W[k, ij] for k in model.ITEM_K]) <= 1
        model.CapacityCons1 = Constraint(model.ITEM_IJ, rule=capacityRule1)

        def capacityRule2(model, k):
            return sum([model.W[k, l] * model.A_l[l] for l in model.ITEM_IJ]) <= model.A_u[k]
        model.CapacityCons2 = Constraint(model.ITEM_K, rule=capacityRule2)

        model.OBJ = Objective(sense=maximize, expr=sum(sum(
            model.W[k, l] * model.j_coeffs[k, l] for k in model.ITEM_K) for l in model.ITEM_IJ))

        opt = SolverFactory(SOLVER)
        opt.solve(model)

        result = np.zeros((N_u, np.sum(N_apps)))

        for k in model.ITEM_K:
            for l in model.ITEM_IJ:
                result[k][l] = value(model.W[k, l]) * \
                    A_labeled[l] / A_unlabeled[k]

        return result

    @staticmethod
    def label_assignement_step_2(N_app, N_u, A_unlabeled, budgets, TargetBudgets):
        """
        Depreciated. No longer used in any part of the code.
        """
        model = ConcreteModel()
        model.N_app = Param(initialize=N_app)
        model.ITEM_I = RangeSet(0, N_app - 1)
        model.ITEM_K = RangeSet(0, N_u - 1)

        def budgets_init(model, k, i):
            return budgets[k][i]
        model.budgets = Param(model.ITEM_K, model.ITEM_I,
                              initialize=budgets_init, within=Reals)

        def A_u_init(model, k):
            return A_unlabeled[k]
        model.A_u = Param(model.ITEM_K, initialize=A_u_init,
                          within=PositiveReals)

        def TargetBudgets_init(model, i):
            return TargetBudgets[i]
        model.Target = Param(
            model.ITEM_I, initialize=TargetBudgets_init, within=PositiveReals)

        model.X = Var(model.ITEM_K, model.ITEM_I, domain=PercentFraction)

        def attributionConsRule(model, k):
            return sum(model.X[k, i] for i in model.ITEM_I) <= 1

        model.attributionCons = Constraint(
            model.ITEM_K, rule=attributionConsRule)

        def CapacityConsRule(model, i):
            return sum([model.A_u[k] * model.budgets[k, i] * model.X[k, i] for k in model.ITEM_K]) / model.Target[i] <= 1

        model.CapacityCons = Constraint(model.ITEM_I, rule=CapacityConsRule)

        model.OBJ = Objective(sense=maximize, expr=sum(sum(
            [model.budgets[k, i] * model.X[k, i] for k in model.ITEM_K]) for i in model.ITEM_I))

        opt = SolverFactory(SOLVER)
        opt.solve(model)

        result = np.zeros((N_u, N_app))
        for k in model.ITEM_K:
            for i in model.ITEM_I:
                result[k][i] = value(model.X[k, i])

        return result
