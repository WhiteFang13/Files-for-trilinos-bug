#include <boost/mpi.hpp>
#include <Tpetra_Core.hpp>
#include "LinearEquationSolver.h"

namespace mpi = boost::mpi;
mpi::communicator world;

int mpiMain(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    return mpiMain(argc, argv);
}

void printThrow(const Teuchos::RCP<const Teuchos::Comm<int>> &comm, const std::ostringstream &errStrm, const std::string &who) {
    std::cerr << who << " threw an exception on one or more processes!" << std::endl;
    for (int r = 0; r < comm->getSize(); ++r) {
        if (r == comm->getRank()) {
            std::cerr << "Process " << comm->getRank() << ": " << errStrm.str() << std::endl;
        }
        comm->barrier(); // wait for output to finish
    }
}

int mpiMain(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);

    Teuchos::RCP<const Teuchos::Comm<int>> comm(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    std::vector<int> ranks;
    for (int rank = 1; rank < comm->getSize(); rank++) {
        ranks.push_back(rank);
    }

    Teuchos::RCP<const Teuchos::Comm<int>> solverComm(comm->createSubcommunicator(ranks));

    if (world.rank() == 0) {
    } else {
        LinearEquationSolver solver(52, solverComm); // Передается число, которое на 1 меньше чем количество уравнений в файле

        solver.readAFromFile("A.txt");
        solver.readBFromFile("B.txt");

        solver.printA();
        solver.printB();

        //----------Solver preconditioner creating--------------------
        int lclSuccessPrec = 1, gblSuccessPrec = 1;
        std::ostringstream errStrmPreconditioner;
        try {
            solver.createPreconditioner();
        } catch (std::exception &e) {
            lclSuccessPrec = 0;
            errStrmPreconditioner << e.what();
        }
        Teuchos::reduceAll<int, int>(*solver.comm(), Teuchos::REDUCE_MIN, lclSuccessPrec, Teuchos::outArg(gblSuccessPrec));
        if (solver.comm()->getRank() == 0) {
            if (gblSuccessPrec != 1) {
                printThrow(solver.comm(), errStrmPreconditioner, "Preconditioner");
            }
        }
        //----------Solver precoditioner creating end--------------------

        //Solve
        int lclSuccessSolve = 1, gblSuccessSolve = 1;
        std::ostringstream errStrmSolver;
        try {
            solver.solve(100, 100000, 1.0e-9);
        } catch (std::exception &e) {
            lclSuccessSolve = 0;
            errStrmSolver << e.what();
        }
        Teuchos::reduceAll<int, int>(*solver.comm(), Teuchos::REDUCE_MIN, lclSuccessSolve, Teuchos::outArg(gblSuccessSolve));
        if (gblSuccessSolve != 1) {
            printThrow(solver.comm(), errStrmSolver, "Solver");
        }
        //Solve
    }

    return 0;
}
