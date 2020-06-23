#include <vector>
#include <omp.h>
#include "nonlinearPropShotsGpu_3D.h"
#include "nonlinearPropGpu_3D.h"

// Constructor
nonlinearPropShotsGpu_3D::nonlinearPropShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector) {

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList_3D();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	assert(getGpuInfo_3D(_gpuList, _info, _deviceNumberInfo));
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;

}

void nonlinearPropShotsGpu_3D::createGpuIdList_3D(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);

	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR [nonlinearPropShotsGpu_3D]: Please provide a list of GPUs to be used ****" << std::endl; assert(1==2);}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR [nonlinearPropShotsGpu_3D]: Please make sure there are no duplicates in the GPU Id list ****" << std::endl; assert(1==2);
		}
	}

	// Check that the user does not ask for more GPUs than shots to be modeled
	if (_nGpu > _nShot){std::cout << "**** ERROR [nonlinearPropShotsGpu_3D]: User required more GPUs than shots to be modeled ****" << std::endl; assert(1==2);}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

// Forward
void nonlinearPropShotsGpu_3D::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(2).n == 1) {
			// std::cout << "Constant source signal over shots" << std::endl;
			constantSrcSignal = 1; }
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {
		// std::cout << "Constant receiver geometry over shots" << std::endl;
		constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create a vector that will contain copies of the model that will be sent to each GPU
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	axis dummyAxis(1);
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), dummyAxis)); // Add a dummy axis so that each component of the model slice vector is a 2D array

    // Create a vector that will contain copies of the data (for each shot) that will be sent to each GPU
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));

    // Create a vector that will contain copies nonlinearPropGpu_3D that will be sent to each GPU
	std::vector<std::shared_ptr<nonlinearPropGpu_3D>> propObjectVector;

	// Initialization for each GPU:
	// (1) Creation of vector of objects, model, and data
	// (2) Memory allocation on GPU

	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu_3D> propGpuObject(new nonlinearPropGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			propGpuObject->getFdParam_3D()->getInfo_3D();

		}

		// Allocate memory on device
		allocateNonlinearGpu_3D(propObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// std::cout << "_nGpu= " << _nGpu << std::endl;
	// std::cout << "_nShot= " << _nShot << std::endl;
	// std::cout << "_gpuList.size() " << _gpuList.size() << std::endl;
	// for (int i=0; i<_gpuList.size(); i++){
	// 	std::cout << "gpuList[" << i << "] = " << _gpuList[i] << std::endl;
	// }

	// Launch nonlinear forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	// #pragma omp parallel for num_threads(2)
	// #pragma omp parallel for
    for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// if (iShot == 1){
		// std::cout << "iShot= " << iShot << " is run by = " << iGpu << " with iGpuId: " << iGpuId << std::endl;
		// std::cout << "iShot= " << iShot << std::endl;
		// std::cout << "iGpuId= " << iGpuId << std::endl;
		// }
			// std::cout << "iShot = " << iShot << std::endl;

		// Copy model slice (wavelet)
		if(constantSrcSignal == 1) {
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(double)*hyperModelSlice->getAxis(1).n);
		} else {
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[iShot*hyperModelSlice->getAxis(1).n]), sizeof(double)*hyperModelSlice->getAxis(1).n);
		}

		// Set acquisition geometry
		if (constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
			// Allocate small evlocity for ginsu
			// Update cuda allocation for propObjectVector[iGpu]
			// Update fd param
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);

		// std::cout << "Before" << std::endl;
		// std::cout << "data" << dataSliceVector[iGpu]->max() << std::endl;
		// Launch modeling
		// std::cout << "[nonlinearShots] Before running forward for iShot= " << iShot << ", iGpu=" << iGpu << std::endl;
		propObjectVector[iGpu]->forward(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);
		// std::cout << "[nonlinearShots] After running forward for iShot= " << iShot << ", iGpu=" << iGpu << std::endl;

		// Store dataSlice into data
		#pragma omp parallel for
        for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
            for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
                (*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
            }
        }
		// std::cout << "After" << std::endl;
		// std::cout << "data" << dataSliceVector[iGpu]->max() << std::endl;
		// std::cout << "[nonlinearShots] Done running forward for iShot= " << iShot << ", iGpu=" << iGpu << std::endl;
    }

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearGpu_3D(iGpu, _gpuList[iGpu]);
	}
}

// Adjoint
void nonlinearPropShotsGpu_3D::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(2).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GP
	axis dummyAxis(1);
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), dummyAxis));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<nonlinearPropGpu_3D>> propObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu_3D> propGpuObject(new nonlinearPropGpu_3D(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			propGpuObject->getFdParam_3D()->getInfo_3D();
		}

		// Allocate memory on device
		allocateNonlinearGpu_3D(propObjectVector[iGpu]->getFdParam_3D()->_vel2Dtw2, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);
		modelSliceVector[iGpu]->scale(0.0); // Initialize each model slice to zero

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch nonlinear adjoint
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if(constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition_3D(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber_3D(iGpu, iGpuId);

		// Launch modeling
		if (constantSrcSignal == 1){
			// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
			propObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			// Copy the shot into model slice --> Is there a way to parallelize this?
			propObjectVector[iGpu]->adjoint(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);
			#pragma omp parallel for
			for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
				(*model->_mat)[iShot][its] += (*modelSliceVector[iGpu]->_mat)[0][its];
			}
		}
	}

	// If using the same wavelet for all shots, stack all shots from all iGpus
	// !!!! Make sure that the parallelization is not done over iGpu !!!!
	if (constantSrcSignal == 1){
		#pragma omp parallel for
		for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
			for (int iGpu=0; iGpu<_nGpu; iGpu++){
				(*model->_mat)[0][its] += (*modelSliceVector[iGpu]->_mat)[0][its];
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearGpu_3D(iGpu, _gpuList[iGpu]);
	}

}
