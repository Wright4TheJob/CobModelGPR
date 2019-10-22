# coverage run unit_test.py
#echo "#################################"
# echo "Coverage:"
# coverage report

# cd ..
echo "-------------"
echo "Documentation"
echo "-------------"
pydocstyle analysis.py
pydocstyle mechanical.py
pydocstyle unit_test.py

echo "----------"
echo "Code Style"
echo "----------"
pylint analysis mechanical unit_test

echo "----------"
echo "Unit Testing"
echo "----------"
coverage run unit_test.py
