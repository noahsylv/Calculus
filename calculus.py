from __future__ import annotations
from functools import reduce
import numpy as np
from numpy.lib.arraysetops import isin
class Derivable:
    def derive(self) -> Derivable:
        return self.derivative().simplify()
    def derivative(self) -> Derivable:
        pass
    def is_zero(self) -> bool:
        pass
    def is_constant(self) -> bool:
        pass
    def simplify(self) -> Derivable:
        return self
    def can_add(self, _) -> bool:
        pass
    def can_multiply(self, _) -> bool:
        pass
    def can_add(self, _) -> bool:
        pass
    def __repr__(self) -> str:
        return str(self)
    def evaluate(self, x):
        return x
    @staticmethod
    def prep(other):
        if type(other) in [float, int]:
            other = Constant(other)
        return other
    def __add__(self, other):
        other = Derivable.prep(other)
        if not isinstance(other, Derivable):
            raise TypeError(f"can't add {type(other)} to Derivable")
        return Sum([self, other])
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other
    def __mul__(self, other):
        other = Derivable.prep(other)
        if not isinstance(other, Derivable):
            raise TypeError(f"can't multiply {type(other)} by Derivable")
        return Product([self, other])
    def __eq__(self, other: object) -> bool:
        return self.simplify().__equals__(Derivable.prep(other).simplify())
    def __truediv__(self, other: object):
        return self * Power(Derivable.prep(other), -1)
    def __rtruediv__ (self, other):
        return other * Power(Derivable.prep(self), -1)
    def __pow__(self, other):
        if type(other) not in [float, int]:
            raise TypeError(f"unsupported ** for Derivable and {type(other)}")
        return Power(self, other)
    def __equals__(self, other):
        return False
    @staticmethod
    def multiply_all(terms_to_combine):
        first = terms_to_combine[0]
        if len(terms_to_combine) == 1:
            return first
        if isinstance(first, Term):
            combined = Term.multiply_terms(terms_to_combine)
            return combined
        # combining functions
        total_product = 0
        for t in terms_to_combine:
            if isinstance(t, Power):
                total_product += t.power
            else: total_product += 1
        if isinstance(first, Power):
            return Power(first.argument, total_product).simplify()
        return Power(first, total_product).simplify()
    @staticmethod
    def add_all(terms_to_combine):
        first = terms_to_combine[0]
        if len(terms_to_combine) == 1:
            return first
        if isinstance(first, Term):
            combined = Term.add_terms(terms_to_combine)
            return combined
        # combining functions
        return len(terms_to_combine) * first

    @staticmethod
    def de_nest(l, class_type, get_sublist = lambda x: x.parts):
        nested_products = [p for p in l if type(p) == class_type]
        if nested_products:
            other_products = [p for p in l if type(p) != class_type]
            nested_parts = [get_sublist(x) for x in nested_products]
            return other_products + reduce(lambda a,b: a + b, nested_parts)
        else:
            return l
    @staticmethod
    def combine_parts(parts, original_num_parts, can_combine_fn, combine_fn, class_type):
        term_sets_to_combine = []
        for part in parts:
            found_set = False
            for t in term_sets_to_combine:
                if can_combine_fn(t[0], part):
                    t.append(part)
                    found_set = True
                    break
            if not found_set:
                term_sets_to_combine.append([part])
        terms_combined = [combine_fn(t) for t in term_sets_to_combine]
        combined = class_type(terms_combined)
        if len(combined.parts) != original_num_parts:
            return combined.simplify()
        return combined

class Product(Derivable):
    def __init__(self, parts) -> None:
        # de-nest if necessary
        self.parts = Derivable.de_nest(parts, Product)
    def evaluate(self, x):
        return reduce(lambda a,b: a*b, [p.evaluate(x) for p in self.parts])
    def __str__(self):
        return str(self.parts[0]) + "".join([f"({str(x)})" for x in self.parts[1:]])
    def derivative(self):
        derivatives = [p.derive().simplify() for p in self.parts]
        out = []
        for i, derived in enumerate(derivatives):
            prod = Product([derived] + [p for j, p in enumerate(self.parts) if i != j])
            out.append(prod)
        return Sum(out).simplify()
    def is_zero(self):
        return any([p.is_zero() for p in self.parts])
    def is_constant(self):
        all_constant = all([p.is_constant() for p in self.parts])
        return self.is_zero() or all_constant
    def can_add(self, other) -> bool:
        return self == other
    def simplify(self) -> Derivable:
        if self.is_zero():
            return Constant(0)
        if len(self.parts) == 1:
            return self.parts[0].simplify()
        num_parts = len(self.parts)
        # get rid of trivial 1's
        new_parts = [p.simplify() for p in self.parts if not (isinstance(p, Term) and p.is_one())]
        if not new_parts:
            return Constant(1)

        # combine terms when possible
        return Derivable.combine_parts(
            new_parts, 
            num_parts, 
            lambda a,b: a.can_multiply(b), 
            lambda t: Derivable.multiply_all(t), 
            Product
        )
    def __equals__(self, other: object) -> bool:
        return compare_parts_objects(self, other, Product)

def compare_parts_objects(a: Derivable, b: Derivable, class_type, get_sublist = lambda x: x.parts):
    if not isinstance(a.simplify(), class_type):
        return a.simplify() == b.simplify()
    if not isinstance(b, class_type):
        return False
    a = a.simplify()
    b = b.simplify()
    if len(get_sublist(a)) != len(b.parts):
        return False
    return all([sum([px == py for px in get_sublist(a)])==1 for py in get_sublist(b)])

class Sum(Derivable):
    def __init__(self, parts) -> None:
        # de-nest
        parts = [Derivable.prep(p).simplify() for p in parts]
        self.parts = Derivable.de_nest(parts, Sum)
    def evaluate(self, x):
        return reduce(lambda a,b: a+b, [p.evaluate(x) for p in self.parts])
    def derivative(self) -> Derivable:
        return Sum(list(map(lambda x: x.derive(), self.parts))).simplify()
    def __get_non_zero_terms(self):
        return [x.simplify() for x in self.parts if not x.is_zero()]
    def __str__(self) -> str:
        non_zero_terms = self.__get_non_zero_terms()
        if non_zero_terms:
            return " + ".join(list(map(str, non_zero_terms)))
        return "0"
    def is_zero(self) -> bool:
        return all([x.is_zero() for x in self.parts])
    def is_constant(self) -> bool:
        return all([x.is_constant() for x in self.parts])
    def __equals__(self, other: object) -> bool:
        return compare_parts_objects(self, other, Sum)
    def simplify(self) -> Derivable:
        if len(self.parts) == 1:
            return self.parts[0].simplify()
        num_parts = len(self.parts)
        # get rid of trivial 0's
        new_parts = [p.simplify() for p in self.parts if not (isinstance(p, Term) and p.is_zero())]
        if not new_parts:
            return Constant(0)
        # combine terms when possible
        return Derivable.combine_parts(
            new_parts, 
            num_parts, 
            lambda a,b: a.can_add(b), 
            lambda t: Derivable.add_all(t), 
            Sum
        )

class Term(Derivable):
    def __init__(self, coef, power) -> None:
        self.coef = coef
        self.power = power if coef != 0 else 0
    def evaluate(self, x):
        return self.coef * x ** self.power
    def __str__(self):
        if self.coef == 0:
            return "0"
        if self.coef == 1:
            if self.power == 0:
                return "1"
            return f"x^{self.power}".replace("x^1","x")
        return f"{self.coef}x^{self.power}".replace("x^0", "").replace("x^1","x")
    def derivative(self):
        return Term(self.coef*self.power, self.power-1)
    def is_zero(self) -> bool:
        return self.coef == 0
    def is_constant(self) -> bool:
        return self.is_zero() or self.power == 0
    def is_one(self) -> bool:
        return self.is_constant() and self.coef == 1
    def can_multiply(self, other) -> bool:
        return Power.power_instance(other, Term)
    def can_add(self, other) -> bool:
        self = self.simplify()
        other = other.simplify()
        if not isinstance(other, Term):
            return False
        return self.power == other.power
    def __equals__(self, other) -> bool:
        if not isinstance(other, Term): return False
        return self.coef == other.coef and self.power == other.power
    @staticmethod
    def multiply_terms(terms) -> Term:
        # assumes terms can be multiplied
        terms = [p.simplify() for p in terms]
        final_coef = reduce(lambda x,y: x*y, [p.coef for p in terms])  
        final_exp = reduce(lambda x,y: x+y, [p.power for p in terms])
        final_term = Term(final_coef, final_exp).simplify()
        return final_term
    @staticmethod
    def add_terms(terms) -> Term:
        # assumes terms can be added
        terms = [p.simplify() for p in terms]
        final_coef = sum([p.coef for p in terms])  
        final_term = Term(final_coef,  terms[0].power).simplify()
        return final_term
class Constant(Term):
    def __init__(self, coef) -> None:
        super().__init__(coef, 0)
    def is_zero(self) -> bool:
        return super().is_zero()
    def is_constant(self) -> bool:
        return super().is_constant()
    def derive(self):
        return super().derive()
class Function(Derivable):
    def __init__(self, argument: Derivable, fn_name = "f") -> None:
        self.argument = argument.simplify()
        self.fn_name = fn_name
    def get_argument(self):
        return self.argument.simplify()
    def derive(self) -> Derivable:
        # chain rule
        coef_deriv = self.get_argument().derive().simplify()
        return Product([self.derive_formula().simplify(), coef_deriv]).simplify()
    def can_multiply(self, other) -> bool:
        if isinstance(other, Power):
            other = other.argument
        curr = self
        if isinstance(curr, Power):
            curr = curr.argument
        return curr == other
    def can_add(self, other) -> bool:
        return self == other
    def __equals__(self, other):
        self = self.simplify()
        other = other.simplify()
        if not isinstance(self, Function):
            return self == other
        return type(self) == type(other) and self.argument == other.argument and self.additional_equals(other)
    def evaluate(self, x):
        return self.function_evaluate(self.get_argument().evaluate(x))
    # -- to be overidden
    def function_evaluate(self, arg):
        return arg
    def additional_equals(self, other):
        return True
    def derive_formula(self):
        return self.get_argument()
    def __str__(self) -> str:
        return f"{self.fn_name}({self.get_argument().__str__()})"
class Cos(Function):
    def __init__(self, argument: Derivable) -> None:
        super().__init__(argument, "cos")
    def derive_formula(self):
        return Product([Constant(-1), Sin(self.get_argument())])
    def function_evaluate(self, arg):
        return np.cos(arg)
class Sin(Function):
    def __init__(self, argument: Derivable) -> None:
        super().__init__(argument, "sin")
    def derive_formula(self):
        return Cos(self.get_argument())
    def function_evaluate(self, arg):
        return np.sin(arg)
class Tan(Function):
    def __init__(self, argument: Derivable) -> None:
        super().__init__(argument, "tan")
    def derive_formula(self):
        a = self.argument
        return Power(Cos(a), -2)
    def function_evaluate(self, arg):
        return np.tan(arg)
def Sec(a) -> Derivable:
    return 1 / Cos(a)
def Csc(a) -> Derivable:
    return 1 / Sin(a)
class Power(Function):
    def __init__(self, argument: Derivable, power: float) -> None:
        super().__init__(argument)
        self.power = power
    def __str__(self) -> str:
        return f"({self.get_argument().__str__()})^{self.power}"
    def derive_formula(self):
        return Product([Constant(self.power), Power(self.get_argument(), self.power - 1).simplify()]).simplify()
    def simplify(self) -> Derivable:
        if self.power == 0:
            return Constant(1)
        if self.power == 1:
            return self.get_argument()
        if self.argument.is_zero():
            return Constant(0)
        if isinstance(self.get_argument(), Term):
            arg = self.get_argument()
            return Term(arg.coef ** self.power, arg.power * self.power)
        return Power(self.get_argument(), self.power)
    def additional_equals(self, other):
        return self.power == other.power
    def function_evaluate(self, a):
        return a ** self.power
    @staticmethod
    def power_instance(x, class_type):
        normal_type =  isinstance(x, class_type)
        power_type = isinstance(x, Power) and isinstance(x.argument, class_type)
        return normal_type or power_type
class X(Term):
    def __init__(self) -> None:
        super().__init__(1, 1)