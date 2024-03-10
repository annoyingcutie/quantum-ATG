import random
import numpy as np
from math import ceil
from scipy.stats import chi2, ncx2
from qiskit import Aer
from qiskit import execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import qiskit.circuit.library as qGate
from qiskit.circuit.gate import Gate
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import standard_errors, ReadoutError


from qiskit.ignis.mitigation.measurement import tensored_meas_cal
from qiskit.extensions import UnitaryGate
from qiskit.ignis.mitigation.measurement import TensoredMeasFitter
from libs_qrem import LeastNormFilter
from qiskit.tools.visualization import plot_histogram


# import sutff
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))

from qatgUtil import *

random.seed(114514)

class QATGConfiguration():
	"""the return results of qatg is described as qatgConfiguration objects"""
	def __init__(self, circuitSetup: dict, simulationSetup: dict, faultObject):
		# circuitSetup: circuitSize, basisGateSet, quantumRegisterName, classicalRegisterName, circuitInitializedStates
		
		self.circuitSize = circuitSetup['circuitSize']
		self.basisGateSet = circuitSetup['basisGateSet']
		self.basisGateSetString = circuitSetup['basisGateSetString']
		self.circuitInitializedStates = circuitSetup['circuitInitializedStates']
		self.backend = Aer.get_backend('qasm_simulator')

		self.oneQubitErrorProb = simulationSetup['oneQubitErrorProb']
		self.twoQubitErrorProb = simulationSetup['twoQubitErrorProb']
		self.zeroReadoutErrorProb = simulationSetup['zeroReadoutErrorProb']
		self.oneReadoutErrorProb = simulationSetup['oneReadoutErrorProb']

		self.targetAlpha = simulationSetup['targetAlpha']
		self.targetBeta = simulationSetup['targetBeta']

		self.simulationShots = simulationSetup['simulationShots']
		self.testSampleTime = simulationSetup['testSampleTime']

		self.faultObject = faultObject

		quantumRegisterName = circuitSetup['quantumRegisterName']
		classicalRegisterName = circuitSetup['classicalRegisterName']
		self.quantumRegister = QuantumRegister(self.circuitSize, quantumRegisterName)
		self.classicalRegister = ClassicalRegister(self.circuitSize, classicalRegisterName)

		self.faultfreeQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT = QuantumCircuit(self.quantumRegister, self.classicalRegister)

		self.faultfreeDistribution = []
		self.faultyDistribution = []
		self.repetition = np.nan
		self.boundary = np.nan
		self.simulatedOverkill = np.nan
		self.simulatedTestescape = np.nan
		self.cktDepth = np.nan
		self.effectSize = np.nan

		self.noiseModel = self.getNoiseModel()
		self.noiseModel_noReadOutError = self.getNoiseModel_noReadOutError()

	def __str__(self):
		rt = ""
		rt += "Target fault: { " + str(self.faultObject) + " }\n"
		rt += "Length: " + str(self.cktDepth)
		rt += "\tRepetition: " + str(self.repetition)
		rt += "\tCost: " + str(self.cktDepth * self.repetition) + "\n"
		rt += "Chi-Value boundary: " + str(self.boundary) + "\n"
		rt += "Effect Size: " + str(self.effectSize) + "\n"
		rt += "Overkill: "+ str(self.simulatedOverkill)
		rt += "\tTest Escape: " + str(self.simulatedTestescape) + "\n"
		# rt += "Circuit: \n" + str(self.faultfreeQCKT)

		return rt

	def getNoiseModel(self):
		# Depolarizing quantum errors
		oneQubitError = standard_errors.depolarizing_error(self.oneQubitErrorProb, 1)
		twoQubitError = standard_errors.depolarizing_error(self.twoQubitErrorProb, 2)
		qubitReadoutError = ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])

		# Add errors to noise model
		noiseModel = NoiseModel()
		noiseModel.add_all_qubit_quantum_error(oneQubitError, self.basisGateSetString)
		noiseModel.add_all_qubit_quantum_error(twoQubitError, ['cx'])
		noiseModel.add_all_qubit_readout_error(qubitReadoutError)

		return noiseModel
    
	def getNoiseModel_noReadOutError(self):
		# Depolarizing quantum errors
		oneQubitError = standard_errors.depolarizing_error(self.oneQubitErrorProb, 1)
		twoQubitError = standard_errors.depolarizing_error(self.twoQubitErrorProb, 2)
		qubitReadoutError = ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])

		# Add errors to noise model
		noiseModel = NoiseModel()
		noiseModel.add_all_qubit_quantum_error(oneQubitError, self.basisGateSetString)
		noiseModel.add_all_qubit_quantum_error(twoQubitError, ['cx'])
		

		return noiseModel

	def setTemplate(self, template, effectSize):
		# template itself is faultfree
		self.effectSize = effectSize

		qbIndexes = self.faultObject.getQubits()

		for gates in template:
			# in template, a list for seperate qubits and a gate for all qubits
			if isinstance(gates, list):
				for k in range(len(gates)):
					self.faultfreeQCKT.append(gates[k], [qbIndexes[k]])
					if self.faultObject.isSameGateType(gates[k]):
						self.faultyQCKT.append(self.faultObject.createFaultyGate(gates[k]), [qbIndexes[k]])
					else:
						self.faultyQCKT.append(gates[k], [qbIndexes[k]])
			elif issubclass(type(gates), Gate):
				self.faultfreeQCKT.append(gates, qbIndexes)
				if self.faultObject.isSameGateType(gates):
					self.faultyQCKT.append(self.faultObject.createFaultyGate(gates), qbIndexes)
				else:
					self.faultyQCKT.append(gates, qbIndexes)
			else:
				raise TypeError(f"Unknown object \"{gates}\" in template")

			for qb in qbIndexes:
				self.faultfreeQCKT.append(qGate.Barrier(qb))
				self.faultyQCKT.append(qGate.Barrier(qb))

		self.faultfreeQCKT.measure(self.quantumRegister, self.classicalRegister)
		self.faultyQCKT.measure(self.quantumRegister, self.classicalRegister)

		self.cktDepth = len(template)

		return

	def simulate(self):
		#simulateJob = execute(self.faultfreeQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		#counts = simulateJob.result().get_counts()
		counts = self.qatg_QREM_faultfreeQCKT()
		self.faultfreeDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultfreeDistribution[int(k, 2)] = counts[k]
		self.faultfreeDistribution = np.array(self.faultfreeDistribution / np.sum(self.faultfreeDistribution))

		#simulateJob = execute(self.faultyQCKT, self.backend, noise_model = self.noiseModel, shots = self.simulationShots)
		#counts = simulateJob.result().get_counts()
		counts = self.qatg_QREM_faultyQCKT()
		self.faultyDistribution = [0] * (2 ** self.circuitSize)
		for k in counts:
			self.faultyDistribution[int(k, 2)] = counts[k]
		self.faultyDistribution = np.array(self.faultyDistribution / np.sum(self.faultyDistribution))

		self.repetition, self.boundary = self.calRepetition()

		self.simulatedOverkill = self.calOverkill()
		self.simulatedTestescape = self.calTestEscape()
		
		return

	def calRepetition(self):
		if self.faultfreeDistribution.shape != self.faultyDistribution.shape:
			raise ValueError('input shape not consistency')

		degreeOfFreedom = self.faultfreeDistribution.shape[0] - 1
		effectSize = qatgCalEffectSize(self.faultyDistribution, self.faultfreeDistribution)
		lowerBoundEffectSize = 0.8 if effectSize > 0.8 else effectSize

		chi2Value = chi2.ppf(self.targetAlpha, degreeOfFreedom)
		repetition = ceil(chi2Value / (lowerBoundEffectSize ** 2))
		nonCentrality = repetition * (effectSize ** 2)
		nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		while nonChi2Value < chi2Value:
			repetition += 1
			nonCentrality += effectSize ** 2
			nonChi2Value = ncx2.ppf(1 - self.targetBeta, degreeOfFreedom, nonCentrality)
		
		boundary = (nonChi2Value * 0.3 + chi2Value * 0.7)
		if repetition >= qatgINT_MAX or repetition <= 0:
			raise ValueError("Error occured calculating repetition")
		
		return repetition, boundary

	def calOverkill(self):
		overkill = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultfreeDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should pass, chiStatistic should > boundary
			if chiStatistic <= self.boundary:
				overkill += 1

		return overkill / self.testSampleTime

	def calTestEscape(self):
		testEscape = 0
		expectedDistribution = self.faultyDistribution
		observedDistribution = self.faultyDistribution

		for _ in range(self.testSampleTime):
			sampledData = random.choices(range(observedDistribution.shape[0]), weights = observedDistribution, k = self.repetition)
			sampledObservedDistribution = np.zeros(observedDistribution.shape[0])
			for d in sampledData:
				sampledObservedDistribution[d] += 1
			sampledObservedDistribution = sampledObservedDistribution / self.repetition

			deltaSquare = np.square(expectedDistribution - sampledObservedDistribution)
			chiStatistic = self.repetition * np.sum(deltaSquare/(expectedDistribution+qatgINT_MIN))

			# test should fail, chiStatistic should <= boundary
			if chiStatistic > self.boundary:
				testEscape += 1

		return testEscape / self.testSampleTime
    
	def qatg_QREM_faultfreeQCKT(self):
    
        #first run the calibration circuit
		n = self.circuitSize
		qr = self.quantumRegister 
		mit_pattern = []
		for i in range(n):
			singlebit = [i]
			mit_pattern.append(singlebit)
    
		meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
        
    
		no_readouterror_job = execute(meas_calibs, backend=self.backend, shots=self.simulationShots,noise_model=self.noiseModel_noReadOutError)
        
		no_readouterror_cal_results = no_readouterror_job .result()
		print("no_readouterror")
		print(no_readouterror_cal_results.get_counts())
        
		noise_job = execute(meas_calibs, backend=self.backend, shots=self.simulationShots, noise_model=self.noiseModel)
		noise_cal_results = noise_job.result()
		noisy_hist = noise_job.result().get_counts()
		print("noisy_hist")
		print(noisy_hist)
        
		meas_fitter = TensoredMeasFitter(noise_cal_results, mit_pattern=mit_pattern)
		print("calibration matrices")
		print(meas_fitter.cal_matrices)
        
		for i in range(n):
			print('Readout fidelity of Q'+str(i)+': %f'%meas_fitter.readout_fidelity(i))
			print('Q'+str(i)+' Calibration Matrix')
			meas_fitter.plot_calibration(i)
        
        
        
		meas_filter = LeastNormFilter(n, meas_fitter.cal_matrices)
        
        
		mitigated_hist = meas_filter.apply(noisy_hist)
		print("mitigated_hist")
		print(mitigated_hist)
        
        #run the fault_free circuit
		faultfree_noreadouterror = execute(self.faultfreeQCKT, backend=self.backend, shots=self.simulationShots,noise_model=self.getNoiseModel_noReadOutError())
		faultfree_noreadouterror_results = faultfree_noreadouterror.result()
        # Results without mitigation
        
		no_readouterror = faultfree_noreadouterror_results.get_counts()
		print("no_readouterror")
		print(no_readouterror)
        
        
        
		faultfree_noisy_job = execute(self.faultfreeQCKT, backend=self.backend, shots=self.simulationShots, noise_model=self.getNoiseModel())
		faultfree_noisy_results = faultfree_noisy_job.result()
        
        # Results without mitigation
		faultfree_raw_counts = faultfree_noisy_results.get_counts()
    
        # Get the filter object
       
        # Results with mitigation
		mitigated_results = meas_filter.apply(faultfree_noisy_results)
		mitigated_counts = mitigated_results.get_counts()
		display(plot_histogram([faultfree_raw_counts, mitigated_counts,no_readouterror], legend=['raw', 'mitigated','no_readouterror']))
		print("raw_counts")
		print(faultfree_raw_counts)
		print("mitigated_counts")
		print(mitigated_counts)
        
		return mitigated_counts
	def qatg_QREM_faultyQCKT(self):
    
        #first run the calibration circuit
		n = self.circuitSize
		qr = self.quantumRegister 
		mit_pattern = []
		for i in range(n):
			singlebit = [i]
			mit_pattern.append(singlebit)
    
		meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
        
    
		no_readouterror_job = execute(meas_calibs, backend=self.backend, shots=self.simulationShots,noise_model=self.noiseModel_noReadOutError)
        
		no_readouterror_cal_results = no_readouterror_job .result()
		print("no_readouterror")
		print(no_readouterror_cal_results.get_counts())
        
		noise_job = execute(meas_calibs, backend=self.backend, shots=self.simulationShots, noise_model=self.noiseModel)
		noise_cal_results = noise_job.result()
		noisy_hist = noise_job.result().get_counts()
		print("noisy_hist")
		print(noisy_hist)
        
		meas_fitter = TensoredMeasFitter(noise_cal_results, mit_pattern=mit_pattern)
		print("calibration matrices")
		print(meas_fitter.cal_matrices)
        
		for i in range(n):
			print('Readout fidelity of Q'+str(i)+': %f'%meas_fitter.readout_fidelity(i))
			print('Q'+str(i)+' Calibration Matrix')
			meas_fitter.plot_calibration(i)
        
        
        
		meas_filter = LeastNormFilter(n, meas_fitter.cal_matrices)
        
        
		mitigated_hist = meas_filter.apply(noisy_hist)
		print("mitigated_hist")
		print(mitigated_hist)
       
        #run the fault_free circuit
		faultfree_noreadouterror = execute(self.faultyQCKT, backend=self.backend, shots=self.simulationShots,noise_model=self.getNoiseModel_noReadOutError())
		faultfree_noreadouterror_results = faultfree_noreadouterror.result()
        # Results without mitigation
        
		no_readouterror = faultfree_noreadouterror_results.get_counts()
		print("no_readouterror")
		print(no_readouterror)
        
        
        
		faultfree_noisy_job = execute(self.faultyQCKT, backend=self.backend, shots=self.simulationShots, noise_model=self.getNoiseModel())
		faultfree_noisy_results = faultfree_noisy_job.result()
        
        # Results without mitigation
		faultfree_raw_counts = faultfree_noisy_results.get_counts()
    
        # Get the filter object
       
        # Results with mitigation
		mitigated_results = meas_filter.apply(faultfree_noisy_results)
		mitigated_counts = mitigated_results.get_counts()
		display(plot_histogram([faultfree_raw_counts, mitigated_counts,no_readouterror], legend=['raw', 'mitigated','no_readouterror']))
		print("raw_counts")
		print(faultfree_raw_counts)
		print("mitigated_counts")
		print(mitigated_counts)
        
		return mitigated_counts

	@property
	def circuit(self):
		return self.faultfreeQCKT
	