# Install Cython SSK

Run

```sh
$ python setup.py build_ext --inplace
```

# Test Cython SSK

Run

```python
>>> import ssk
>>> ssk.ssk('car', 'car', n=2, lam=0.5)
1.0
```

