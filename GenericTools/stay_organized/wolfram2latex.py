def wolfram2latex(string):
    new_string = string.replace('Sqrt', '\sqrt').replace('[', '{').replace(']', '}')
    new_string = new_string.replace('Pi', '\pi').replace('E', 'e').replace('I', 'i')

    return new_string



string = '-(E^(I f x - x^2/2) (Sqrt[2/Pi] ((-2 I) f + x) + E^(x^2/2) f^2 Erf[x/Sqrt[2]]))'

ns = wolfram2latex(string)

print(ns)
