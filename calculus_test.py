from calculus import *

def exhaust(eqn):
    if not eqn.is_zero():
        return exhaust(eqn.derive())
    return eqn

def test_exhaust(f, should_exhaust):
    try:
        assert exhaust(f) and should_exhaust
    except RecursionError:
        assert not should_exhaust

def num_derive(eqn, num):
    for _ in range(num):
        eqn = eqn.derive()
    return eqn

def test_simple_polynomial():
    for slope in [-2, -1, 0, 1, 2]:
        for power in [0, 1, 2, 3.5]:
            function = Term(slope, power)
            actual = function.derive()
            expected = Term(power*slope, power-1)
            assert actual == expected

def test_power_simplification():
    for function_gen in [lambda: X(), lambda: Sin(Power(X(), -1))]:
        actual = Power(function_gen(), 1).simplify()
        expected = function_gen()
        assert actual == expected

def test_different_argument_functions_not_equal():
    assert Sin(Term(1,2)) != Sin(Term(1,3))
    assert Sin(X()) != Sin(Term(1,3))

def test_functions_equal():
    assert Sin(X()) == Sin(Term(1,1))
    assert Sin(X()) == Power(Sin(Term(1,1)),1)
    assert Power(X(), 2) == X() * X()
    assert Power(X(), 2) * X() * X() == Power(Power(X(), 2), 2) 

def test_trig_functions():
    assert Sin(X()).derive() == Cos(X())
    assert Sin(Term(1,2)).derive() == Product([Term(2,1), Cos(Term(1,2))])
    assert Sec(X()) == Power(Cos(X()), -1)
    assert Csc(X()) == Power(Sin(X()), -1)    

def test_multiplication():
    assert X() * X() == Term(1,2)
    assert 0 * X() == 0
    assert X() * Sin(X()) == Sin(X()) * X()
    assert X() * X() != Term(1,3)
    assert X() * Sin(X()) != Sin(X()) * X() * X()
    assert 3 * X() == Term(3,1)
    assert X() * 3 == Term(3,1)
    assert 3 * X() != Term(2,1)
    assert X() * 3 != Term(2,1)
    assert X() * 3 == 3 * X()
    assert 3 * X() == X() * 3
    
def test_nested_multiplication():
    assert X() * Product([Sin(X()), Cos(X())]) == Product([X(), Sin(X()), Cos(X())])
    assert Product([X(), Product([Sin(X()), Cos(X())])]) == Product([X(), Sin(X()), Cos(X())])
    assert Constant(1) * Product([Sin(X()), Cos(X())]) != Product([X(), Sin(X()), Cos(X())])
    assert Product([Constant(1), Product([Sin(X()), Cos(X())])]) != Product([X(), Sin(X()), Cos(X())])
    
def test_addition():
    assert X() + X() == Term(2,1)
    assert X() + X() != Term(3,1)
    assert 3 + X() == X() + 3
    assert 3 + X() != X() + 2
    assert Sin(X()) + Sin(X()) == 2 * Sin(X())
    assert Sin(X()) + 5 + Sin(X()) == 2 * Sin(X()) + 2 + 3 + 0
    assert Sin(X()) + 5 + Sin(X()) != 2 * Sin(X()) + 2 + 3 + 1
    assert 3 * X() + X() == 4 * X()
    assert 1 + (2 + Constant(3)) == 6
    assert (X() * Sin(X())) + (X() * Sin(X())) == (X() * Sin(X())) * 2
    assert X() == X() + 0
    assert 3 * X() + 2 * X() * X() + X() + Term(1,2) == Term(4, 1) + Term(3, 2) 
    assert 3 * X() + 2 * X() * X() + X() + Term(2,2) != Term(4, 1) + Term(3, 2) 

def test_nested_addition():
    assert Sum([1,Constant(2) + X()]) == 3 + X()
    assert Sum([Cos(X()), Sum([Sin(X()), Tan(X())])]) == Sum([Sin(X()), Tan(X()), Cos(X())])

def test_division():
    assert X() / 5 == X() * (1/5)
    assert X() / .2 == 5 * X()
    assert X() / Sin(X()) == X() * Power(Sin(X()), -1)

def test_power():
    assert Power(X(), 2) == X() * X()
    assert Power(X(), 2) != X() * X() * X()
    assert Power(Sin(X()), 2) == Sin(X()) * Sin(X())
    assert Power(Sin(X()), 3) != Power(Sin(X()), 2)

def test_chain_rule():
    assert Cos(X() * X()).derive() == -2 * X() * Sin(X() * X())
    assert Cos(Sin(2 * X())).derive() == -2 * Sin(Sin(2 * X())) * Cos(2 * X())
    assert Power(Cos(X()),2).derive() == -2 * Cos(X()) * Sin(X())

def test_product_rule():
    assert (X() * Sin(X())).derive() == Sin(X()) + X() * Cos(X())
    assert (X() * Sin(X()) * Cos(X())).derive() == X().derive() * Sin(X()) * Cos(X()) \
                                                + X() * Sin(X()).derive() * Cos(X()) \
                                                + X() * Sin(X()) * Cos(X()).derive()

def test_infinite_derivative():
    test_exhaust(Sin(X()), False)
    test_exhaust(X(), True)
    test_exhaust(X() * X() * X() * X(), True)
    test_exhaust(Power(X(), 15), True)

def test_multiple_derivative():
    assert num_derive(X() * X() * X() * 5, 2) == 30 * X()

def assert_close(a, b, delta = 0.0000001):
    assert abs(a-b) < delta

def assert_not_close(a, b, delta = 0.0000001):
    assert abs(a-b) > delta

def test_evaluate():
    for x in [-2, -.2, 0, 1, 1.4]:
        assert_close(X().evaluate(x), x)
        assert_close((3 * X() * X()).evaluate(x), 3 * x * x)
        if x not in [0, 1]:
            assert_not_close((3 * X() * X()).evaluate(x), 3 * x * x * x)
        assert_close((3 + X() + X()).evaluate(x), 3 + x + x)
        if x != 0:
            assert_not_close((3 + X() + X()).evaluate(x), 3 + x + x + x)
        assert_not_close((3 + X() + X()).evaluate(x), 2 + x + x)
        assert_close(Sin(X() * X() * 2 + 3).evaluate(x), np.sin(x * x * 2 + 3))
        assert_close(Cos(X() * X() * 4 + 1).evaluate(x), np.cos(x * x * 4 + 1))
        assert_close(Tan(X() * X() * 4 + 1).evaluate(x), np.tan(x * x * 4 + 1))
        assert_close(Power(X() + 3, .7).evaluate(x), (x + 3) ** 0.7)
        assert_close(Power(Sin(X() * X()), 3).evaluate(x), np.sin(x * x) ** 3)
        assert_close(Tan(X()).evaluate(x), (Sin(X()) / Cos(X())).evaluate(x))
        assert_not_close(Sin(X()).evaluate(x), Cos(X()).evaluate(x))

def test_power_operator():
    assert 3 * X() ** 2 + 1 == 3 * X() * X() + 1
    assert 3 * X() ** 2 + 1 != 2 * X() * X() + 1
    assert 3 * X() ** 2 + 1 != 3 * X() * X()
    assert 3 * X() ** 1.5 + 1 != 3 * X() ** 2 + 1
    assert Sin(X()) ** 1.5 == Power(Sin(X()), 1.5)
    
def test_simplify():
    assert isinstance(Product([X(), Constant(1)]).simplify(), Term)
    assert isinstance(Sum([X(), Constant(0)]).simplify(), Term)
    
def test():
    test_simple_polynomial()
    test_power_simplification()
    test_different_argument_functions_not_equal()
    test_trig_functions()
    test_functions_equal()
    test_multiplication()
    test_nested_multiplication()
    test_addition()
    test_nested_addition()
    test_division()
    test_power()
    test_chain_rule()
    test_product_rule()
    test_infinite_derivative()
    test_multiple_derivative()
    test_evaluate()
    test_power_operator()
    test_simplify()
    print("all tests pass")

test()
