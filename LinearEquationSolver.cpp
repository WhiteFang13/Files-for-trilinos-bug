#include "LinearEquationSolver.h"
#include <fstream>
#include <Teuchos_StringIndexedOrderedValueObjectContainer.hpp>

LinearEquationSolver::LinearEquationSolver(size_t numGlobalElements, Teuchos::RCP<const Teuchos::Comm<int>> &solverComm): solverComm(solverComm),
                                                                                                                          numGlobalElements((GO)numGlobalElements + 1),
                                                                                                                          map(new Tpetra::Map<LO, GO, node_type>(
                                                                                                                                  numGlobalElements + 1, 0, solverComm)),
                                                                                                                          A(map, numGlobalElements + 1), //TODO: not sure in 2 par
                                                                                                                          b(map), x(map, true),
                                                                                                                          additiveSchwarz(rcpFromRef(A)) {
}

void LinearEquationSolver::createPreconditioner() {
    // Set RILUK parameters
    Teuchos::ParameterList innerPlist;
    innerPlist.set("fact: drop tolerance", 0.0);
    innerPlist.set("fact: iluk level-of-fill", 1);
    innerPlist.set("fact: relax value", 0.0);

    // Set outer preconditioner parameters
    Teuchos::ParameterList ASlist;
    ASlist.set("inner preconditioner name", "RILUK");
    ASlist.set("inner preconditioner parameters", innerPlist);
    ASlist.set("schwarz: combine mode", "ZERO");
    ASlist.set("schwarz: overlap level", 1); //warning: I don't know why sometimes it crashes all program, but it does
    ASlist.set("schwarz: use reordering", false);

    additiveSchwarz.setParameters(ASlist);

    // Compute (set up) the (outer) preconditioner
    additiveSchwarz.initialize();
    additiveSchwarz.compute();
}

void LinearEquationSolver::solve(const int numBlocks, const int maxIters, const double tol) {
    // Set GMRES (iterative linear solver) parameters
    Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
    solverParams->set("Num Blocks", numBlocks);
    solverParams->set("Block Size", 1);
    solverParams->set("Maximum Iterations", maxIters);
    solverParams->set("Maximum Restarts", 10);
    solverParams->set("Convergence Tolerance", tol);
    solverParams->set("Implicit Residual Scaling", "None");
    solverParams->set("Explicit Residual Scaling", "None");

    // Create the GMRES solver using a "factory" and
    // the list of solver parameters created above.
    typedef Belos::SolverFactory<ST, Tpetra::MultiVector<ST, LO, GO, node_type>, Tpetra::Operator<ST, LO, GO, node_type>> belos_factory_type;
    typedef Belos::SolverManager<ST, Tpetra::MultiVector<ST, LO, GO, node_type>, Tpetra::Operator<ST, LO, GO, node_type>> solver_type;
    Teuchos::RCP<solver_type> solver = belos_factory_type().create("GMRES", solverParams);

    // Create a LinearProblem struct with the problem to solve.
    // A, X, B, and M are passed by (smart) pointer, not copied.
    typedef Belos::LinearProblem<ST, Tpetra::MultiVector<ST, LO, GO, node_type>, Tpetra::Operator<ST, LO, GO, node_type>> problem_type;

    Teuchos::RCP<problem_type> problem(new problem_type(rcpFromRef(A), rcpFromRef(x), rcpFromRef(b)));
    problem->setRightPrec(rcpFromRef(additiveSchwarz));

    // Tell the solver what problem you want to solve.
    solver->setProblem(problem);
    solver->reset(Belos::Problem);

    Belos::ReturnType result = solver->solve();

    if (solverComm->getRank() == 0) {
        if (solver->isLOADetected()) {
            std::cout << "Detected a loss of accuracy!" << std::endl;
        }
        std::cout << "Iteration's num: " << solver->getNumIters() << std::endl;
        if (result == Belos::Converged) {
            std::cout << "Result converged." << std::endl;
        } else {
            std::cout << "Result did not converged." << std::endl;
        }
        std::cout << "It achieved a tolerance of: " << solver->achievedTol() << std::endl;
    }
}

std::vector<double> LinearEquationSolver::getX() {
    std::vector<double> values(map->getNodeNumElements());

    const auto &view = x.get1dView();
    for (auto i = 0; i < map->getNodeNumElements(); i++) {
        values[i] = view[i];
    }

    return values;
}

void LinearEquationSolver::printA() {
    for (auto i = 0; i < A.getRowMap()->getNodeNumElements(); i++) {
        auto rowNum = A.getRowMap()->getGlobalElement(i);
        auto numEntriesInRow = A.getNumEntriesInGlobalRow(rowNum);

        Teuchos::Array<ST> rowvals(numEntriesInRow);
        Teuchos::Array<GO> rowinds(numEntriesInRow);
        std::cout << "Row " << rowNum << ": ";
        A.getGlobalRowCopy(rowNum, rowinds, rowvals, numEntriesInRow);
        for (auto j = 0; j < numEntriesInRow; j++) {
            std::cout << " [" << rowinds[j] << "] = " << rowvals[j];
        }
        std::cout << std::endl;
    }
}

void LinearEquationSolver::printB() {
    for (auto i = 0; i < map->getNodeNumElements(); i++) {
        std::cout << map->getGlobalElement(i) << ") " << b.get1dView()[i] << std::endl;
    }
}

void LinearEquationSolver::saveAToFile(const std::string &fileName) {
    std::ofstream file(fileName);

    for (GO rowNum = 0; rowNum < numGlobalElements; rowNum++) {
        auto numEntriesInRow = A.getNumEntriesInGlobalRow(rowNum);

        Teuchos::Array<ST> rowvals(numEntriesInRow);
        Teuchos::Array<GO> rowinds(numEntriesInRow);
        A.getGlobalRowCopy(rowNum, rowinds, rowvals, numEntriesInRow);

        for (auto j = 0; j < numEntriesInRow; j++) {
            file  << " " << rowinds[j] << " " << rowvals[j];
        }
        file << '\n';
    }
}

void LinearEquationSolver::saveBToFile(const std::string &fileName) {
    std::ofstream file(fileName);

    for (GO rowNum = 0; rowNum < numGlobalElements; rowNum++) {
        file << b.get1dView()[rowNum] << '\n';
    }
}

void LinearEquationSolver::readAFromFile(const std::string &fileName) {
    std::ifstream file(fileName);

    A.resumeFill();
    for (GO rowNum = 0; rowNum < numGlobalElements; rowNum++) {
        Teuchos::Array<GO> indexes;
        Teuchos::Array<ST> values;

        while (file.peek() != '\n') {
            GO index;
            file >> index;
            indexes.push_back(index);

            ST value;
            file >> value;
            values.push_back(value);
        }
        file.get();

        if (map->isNodeGlobalElement(rowNum)) {
            A.insertGlobalValues(rowNum, indexes, values);
        }
    }
    A.fillComplete();
}

void LinearEquationSolver::readBFromFile(const std::string &fileName) {
    std::ifstream file(fileName);

    for (GO rowNum = 0; rowNum < numGlobalElements; rowNum++) {
        ST value;
        file >> value;

        if (map->isNodeGlobalElement(rowNum)) {
            b.replaceGlobalValue(rowNum, value);
        }
    }
}
