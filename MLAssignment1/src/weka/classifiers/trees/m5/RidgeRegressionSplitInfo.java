/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * RidgeRegressionSplitInfo.java
 * Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.m5;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import no.uib.cipr.matrix.*;
import no.uib.cipr.matrix.Matrix;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.*;
import weka.experiment.PairedStats;

/**
 * Finds split points using correlation.
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 10169 $
 */
public final class RidgeRegressionSplitInfo implements Cloneable, Serializable,
        SplitEvaluate, RevisionHandler {

    /** for serialization */
    private static final long serialVersionUID = 4212734895125452770L;

    private int m_position;

    /**
     * the maximum impurity reduction
     */
    private double m_maxImpurity;

    /**
     * the attribute being tested
     */
    private int m_splitAttr;

    /**
     * the best value on which to split
     */
    private double m_splitValue;

    /**
     * the number of instances
     */
    private int m_number;

    /**
     * Constructs an object which contains the split information
     *
     * @param low the index of the first instance
     * @param high the index of the last instance
     * @param attr an attribute
     */
    public RidgeRegressionSplitInfo(int low, int high, int attr) {
        initialize(low, high, attr);
    }

    /**
     * Makes a copy of this RidgeRegressionSplitInfo object
     */
    @Override
    public final SplitEvaluate copy() throws Exception {
        RidgeRegressionSplitInfo s = (RidgeRegressionSplitInfo) this.clone();

        return s;
    }

    /**
     * Resets the object of split information
     *
     * @param low the index of the first instance
     * @param high the index of the last instance
     * @param attr the attribute
     */
    public final void initialize(int low, int high, int attr) {
        m_number = high - low + 1;
        m_position = -1;
        m_maxImpurity = -Double.MAX_VALUE;
        m_splitAttr = attr;
        m_splitValue = 0.0;
    }

    /**
     * Finds the best splitting point for an attribute in the instances
     *
     * @param attr the splitting attribute
     * @param insts the instances
     * @exception Exception if something goes wrong
     */
    // Method to alter
    @Override
    public final void attrSplit(int attr, Instances insts) throws Exception {
        //System.out.println("Using My Code");
        //int i;
        int len;
        int low = 0;
        int high = insts.numInstances() - 1;
        PairedStats full = new PairedStats(0.01);
        //PairedStats leftSubset = new PairedStats(0.01);
        //PairedStats rightSubset = new PairedStats(0.01);
        int classIndex = insts.classIndex();
        double leftCorr, rightCorr;
        double leftVar, rightVar, allVar;
        double order = 2.0;
        double ridge = 0.01;

        initialize(low, high, attr);

        if (m_number < 4) {
            return;
        }

        //len = ((high - low + 1) < 5) ? 1 : (high - low + 1) / 5;
        len = (m_number < 5) ? 1 : m_number / 5;
        m_position = low;

        double sumYSquared = 0;
        for (int i = low; i <= high; i++) {
            sumYSquared += insts.instance(i).classValue() * insts.instance(i).classValue();
        }

        // Step 1
        Instances leftSubset = new Instances(insts, low, len);
        Instances rightSubset = new Instances(insts, len, m_number - len);

        // Step 4
        Matrix SL1 = getS(leftSubset);
        // Step 5
        Matrix SR1 = getS(rightSubset);
        // Step 6
        Matrix GL1 = getG(leftSubset);
        // Step 7
        Matrix GR1 = getG(rightSubset);
        // Step 2
        Matrix AL1 = getA(leftSubset, ridge);
        // Step 3
        Matrix AR1 = getA(rightSubset, ridge);
        // Step 8
        double RSS = getRSS(AL1, GL1, SL1) + getRSS(AR1, GR1, SR1) + sumYSquared;

        //System.out.println("CODE IS WORKING!!");

        //Step 9
        Matrix prevAL = AL1;
        Matrix prevAR = AR1;
        Matrix prevSL = SL1;
        Matrix prevSR = SR1;
        Matrix prevGL = GL1;
        Matrix prevGR = GR1;

        // Create appropriately configured multiple linear regression object that we will clone
        LinearRegression configuredRegressionModel = new LinearRegression();
        configuredRegressionModel.setRidge(ridge);
        configuredRegressionModel.setAttributeSelectionMethod(new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
        configuredRegressionModel.setEliminateColinearAttributes(false);
        configuredRegressionModel.turnChecksOff();

        for (int i = low + len; i <= high - len - 1; i++) {
            Instances curr = new Instances(insts, i, 1);
            //Step 10
            Matrix ALK = Update_A_Inv(prevAL, curr, true);
            //Step 11
            Matrix ARK = Update_A_Inv(prevAR, curr, false);
            //Step 12
            Matrix SLK = prevSL.add(getS(curr));
            //Step 13
            Matrix SRK = prevSR.add(getS(curr).scale(-1));
            //Step 14
            Matrix GLK = prevGL.add(getG(curr));
            //Step 15
            Matrix GRK = prevGR.add(getG(curr).scale(-1));
            //Step 15
            RSS = getRSS(ALK, GLK, SLK) + getRSS(ARK, GRK, SRK) + sumYSquared;

            prevAL = ALK;
            prevAR = ARK;
            prevSL = SLK;
            prevSR = SRK;
            prevGL = GLK;
            prevGR = GRK;

            // Naive Implementation
            Instance currentInstance = insts.instance(i);
            Instance nextInstance = insts.instance(i + 1);
            leftSubset.add(currentInstance);
            rightSubset.remove(0); // This is super inefficient

            double splitCandidate = (currentInstance.value(attr) + nextInstance.value(attr)) * 0.5;
            if (splitCandidate < nextInstance.value(attr)) {

                LinearRegression leftRegressionModel = (LinearRegression) AbstractClassifier.makeCopy(configuredRegressionModel);
                LinearRegression rightRegressionModel = (LinearRegression)AbstractClassifier.makeCopy(configuredRegressionModel);
                leftRegressionModel.buildClassifier(leftSubset);
                rightRegressionModel.buildClassifier(rightSubset);

                Evaluation leftEvaluation = new Evaluation(leftSubset);
                leftEvaluation.evaluateModel(leftRegressionModel, leftSubset);
                double leftRMSE = leftEvaluation.rootMeanSquaredError();
                Evaluation rightEvaluation = new Evaluation(rightSubset);
                rightEvaluation.evaluateModel(rightRegressionModel, rightSubset);
                double rightRMSE = rightEvaluation.rootMeanSquaredError();

                double currentRSS = leftSubset.numInstances() * (leftRMSE * leftRMSE) +
                        rightSubset.numInstances() * (rightRMSE * rightRMSE);

                //System.out.println("Naive: " + currentRSS + "\tFast:  " + RSS);

                if (-currentRSS > m_maxImpurity) {
                    m_maxImpurity = -currentRSS;
                    m_splitValue = splitCandidate;
                    m_position = i;
                }
            }
        }
    }

    public Matrix getA(Instances inst, double ridge) {
        DenseMatrix A = new DenseMatrix(inst.numAttributes(),inst.numAttributes());
        Matrix X_1timesX_1T = getG(inst);
        int d = inst.numAttributes() - 1;
        DenseMatrix Id = Matrices.identity(d+1);
        Id.set(d, d, 0);
        Id.scale(ridge);
        Matrix curr = X_1timesX_1T.add(Id);
        A.add(curr);
        DenseMatrix I = Matrices.identity(A.numRows());
        DenseMatrix AI = I.copy();
        return A.solve(I, AI);
    }

    public Matrix getS(Instances inst) {
        DenseMatrix S = new DenseMatrix(inst.numAttributes(),1);
        for (int i=0; i < inst.numInstances(); i++) {
            DenseMatrix X_i = new DenseMatrix(inst.numAttributes(), 1);
            int index = 0;
            Instance insts = inst.instance(i);
            for (int j = 0; j < insts.numAttributes(); j++) {
                if (j != inst.classIndex()) {
                    X_i.set(index++, 0, insts.value(j));
                }
            }
            X_i.set(index++, 0, 1);
            double Yi = 0;
            insts = inst.instance(i);
            for (int j = 0; j < insts.numAttributes(); j++) {
                if (j == inst.classIndex()) {
                    Yi = insts.value(j);
                }
            }
            X_i.scale(Yi);
            S.add(X_i);
        }

        return S;
    }

    public Matrix getG(Instances inst) {
        DenseMatrix G = new DenseMatrix(inst.numAttributes(),inst.numAttributes());
        for (int i=0; i < inst.numInstances(); i++) {
            DenseMatrix X_i = new DenseMatrix(inst.numAttributes(), 1);
            int index = 0;
            Instance insts = inst.instance(i);
            for (int j = 0; j < insts.numAttributes(); j++) {
                if (j != inst.classIndex()) {
                    X_i.set(index++, 0, insts.value(j));
                }
            }
            X_i.set(index++, 0, 1);
            G.add(new DenseMatrix(inst.numAttributes(),inst.numAttributes()).rank1(X_i));
        }

        return G;
    }

    public double getRSS(Matrix A, Matrix G, Matrix S) {
        Matrix Part1 = new DenseMatrix(1, S.numRows());
        Part1 = S.transpose(new DenseMatrix(1, S.numRows())).mult(A, Part1);
        Part1 = Part1.mult(G, new DenseMatrix(1,S.numRows()));
        Part1 = Part1.mult(A, new DenseMatrix(1, S.numRows()));
        Part1 = Part1.mult(S, new DenseMatrix(1, 1));

        Matrix Part2 = new DenseMatrix(1, S.numRows());
        Part2 = S.transpose(Part2);
        Part2 = Part2.mult(A, new DenseMatrix(1, S.numRows()));
        Part2 = Part2.mult(S, new DenseMatrix(1,1));

        return Part1.get(0,0) - Part2.get(0,0) * 2;
    }

    public Matrix Update_A_Inv(Matrix AKX, Instances X, boolean leftNode) {
        //Algorithm 3
        //Step 2
        DenseMatrix Xk = new DenseMatrix(X.numAttributes(), 1);
        int index = 0;
        Instance insts = X.instance(0);
        for (int j = 0; j < insts.numAttributes(); j++) {
            if (j != X.classIndex()) {
                Xk.set(index++, 0, insts.value(j));
            }
        }
        Xk.set(index++, 0, 1);

        Matrix Zk = AKX.mult(Xk, new DenseMatrix(X.numAttributes(), 1));
        Matrix top = Zk.mult(Zk.transpose(new DenseMatrix(1, X.numAttributes())), new DenseMatrix(X.numAttributes(),X.numAttributes()));
        double bottom = Xk.transpose(new DenseMatrix(1, X.numAttributes())).mult(Zk, new DenseMatrix(1,1)).get(0,0);
        //Step 3
        if (leftNode == true) {
            //Step 4
            top = top.scale(-1);
            bottom = 1 + bottom;
        }
        //Step 5
        else {
            //Step 6
            bottom = 1 - bottom;

        }

        //Step 7
        Matrix Gk = top.scale(1/bottom);
        return AKX.add(Gk);
    }

    /**
     * Returns the impurity of this split
     *
     * @return the impurity of this split
     */
    @Override
    public double maxImpurity() {
        return m_maxImpurity;
    }

    /**
     * Returns the attribute used in this split
     *
     * @return the attribute used in this split
     */
    @Override
    public int splitAttr() {
        return m_splitAttr;
    }

    /**
     * Returns the position of the split in the sorted values. -1 indicates that a
     * split could not be found.
     *
     * @return an <code>int</code> value
     */
    @Override
    public int position() {
        return m_position;
    }

    /**
     * Returns the split value
     *
     * @return the split value
     */
    @Override
    public double splitValue() {
        return m_splitValue;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10169 $");
    }
}