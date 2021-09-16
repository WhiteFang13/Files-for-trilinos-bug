#ifndef LINEAREQUATIONSOLVER_H
#define LINEAREQUATIONSOLVER_H

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Ifpack2_AdditiveSchwarz.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

typedef Tpetra::Vector<>::scalar_type ST;
typedef Tpetra::Vector<>::local_ordinal_type LO;
typedef Tpetra::Vector<>::global_ordinal_type GO;
typedef Tpetra::Vector<>::node_type node_type;
typedef Ifpack2::AdditiveSchwarz<Tpetra::RowMatrix<ST, LO, GO, node_type>> AdditiveSchwarz;

class LinearEquationSolver {
    Teuchos::RCP<const Teuchos::Comm<int>> &solverComm; // Communicator

    const GO numGlobalElements; // The number of rows and columns in the matrix

    Teuchos::RCP<const Tpetra::Map<LO, GO, node_type>> map; // Matrix's row Map.

    Tpetra::CrsMatrix<ST, LO, GO, node_type> A; //Matrix A
    Tpetra::Vector<ST, LO, GO, node_type> b; //Vector B
    Tpetra::Vector<ST, LO, GO, node_type> x; //vector X

    AdditiveSchwarz additiveSchwarz; // Preconditioner

public:
    LinearEquationSolver(size_t numGlobalElements, Teuchos::RCP<const Teuchos::Comm<int>> &comm);

    void createPreconditioner();

    void solve(int numBlocks, int maxIters, double tol);

    void printA();

    void printB();

    std::vector<double> getX();

    Teuchos::RCP<const Teuchos::Comm<int>> &comm() {
        return solverComm;
    }

    //SUPPORTING TEMPORARY FUNCTIONS

    void saveAToFile(const std::string &fileName); //ONLY FOR 1 SOLVER THREAD
    void saveBToFile(const std::string &fileName); //ONLY FOR 1 SOLVER THREAD

    void readAFromFile(const std::string &fileName);
    void readBFromFile(const std::string &fileName);
};

#endif //LINEAREQUATIONSOLVER_H