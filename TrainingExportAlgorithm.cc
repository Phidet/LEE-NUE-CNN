/** 
*  @file   larpandoracontent/LArWorkshop/TrainingExportAlgorithm.cc 
* 
*  @brief  Implementation of the TrainingExport algorithm class. 
* 
*  $Log: $ */
#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/MyArea/TrainingExportAlgorithm.h"
#include "PandoraMonitoringApi.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArInteractionTypeHelper.h"
#include "larpandoracontent/LArHelpers/LArMonitoringHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>
#include <array>

//#include "larpandora/LArPandoraInterface/LArPandoraGeometry.h"

using namespace pandora;

namespace lar_content
{
	StatusCode TrainingExportAlgorithm::Run()
	{		
		const CaloHitList *pCaloHitListU(nullptr);
		const CaloHitList *pCaloHitListV(nullptr);
		const CaloHitList *pCaloHitListW(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameV, pCaloHitListV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameW, pCaloHitListW));

		CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
		CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
		CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());

		const PfoList *pPfoList(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));
		
		float minX(0);
		float minZ(0);
		OneShowerMinBoundaries(pPfoList, minX, minZ);
		// float minX, minZ;
		// if(OneShowerMinBoundaries(pPfoList, minX, minZ))
		// {
		// }
			PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorU, minX, minZ));
		return STATUS_CODE_SUCCESS;
	}

	bool TrainingExportAlgorithm::OneShowerMinBoundaries(const PfoList *const pPfoList, float &minX, float &minZ)
	{
		PfoList pfoListCrop;
		//const ParticleFlowObject *pShowerPfo(nullptr);
		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if (LArPfoHelper::IsShower(pPfo)){ //&& LArPfoHelper::IsNeutrinoFinalState(pPfo)
	    		if(!pfoListCrop.empty()) // If more than one shower is found
	    		{
	    			return false;
	    		}
	    		pfoListCrop.push_back(pPfo);
	    	}
	    }
	    if(pfoListCrop.empty()) // If more than one shower is found
	    {
	    	return false;
	    }
		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds tracks close to shower to pfoListCrop
		{
			if (LArPfoHelper::IsTrack(pPfo) && LArPfoHelper::IsNeutrinoFinalState(pPfo) && LArPfoHelper::GetThreeDSeparation(pPfo, pfoListCrop.front())<30){
				pfoListCrop.push_back(pPfo);
			}
		}
		std::cout<<"------------------ Point T1.11"<<std::endl;
		CaloHitList caloHitListInCrop;
		std::cout<<"------------------ Point T1.11.1"<<std::endl;
		LArPfoHelper::GetCaloHits(pfoListCrop, TPC_VIEW_U, caloHitListInCrop);
		std::cout<<"------------------ Point T1.12"<<std::endl;
		CartesianVector v =  LArPfoHelper::GetVertex(pfoListCrop.front())->GetPosition();
		v = LArGeometryHelper::ProjectPosition(this->GetPandora(), v, TPC_VIEW_U); // Project 3D vertex onto 2D U view
		const float vertexX = v.GetX();
	    minZ = v.GetZ()-5.0; // takes vertex z position as shower start
	    const int seg = 128;
	    std::array<uint, seg>  hitXDensity= {0}; // Always combining 8 wires
	    for (const CaloHit *const pCaloHit : caloHitListInCrop)
	    {
	    	const float x = pCaloHit->GetPositionVector().GetX();
	    	const float z = pCaloHit->GetPositionVector().GetZ();
	    	const int pixelX = (int) ((x-vertexX)/0.3 + IMSIZE)/(2*IMSIZE/seg);
	    	if(pixelX>=0 && pixelX<seg && (z-minZ)/0.3<IMSIZE && (z-minZ)/0.3>=0)
	    		hitXDensity[pixelX]++;
	    }
	    std::cout<<"------------------ Point T1.14"<<std::endl;
	    int left(0);
	    int right(seg/2);
	    int loopCounter(1);
	    int leftTotal, rightTotal;
	    do{
	    	leftTotal = 0;
	    	rightTotal = 0;
	    	for(int i=0; i<seg/2; i++){
	    		leftTotal += hitXDensity[left+i];
	    		rightTotal += hitXDensity[right+i];
	    	}

	    	if(leftTotal>rightTotal) right -= (seg/4)/loopCounter;
	    	else left += (seg/4)/loopCounter;
	    	loopCounter *=2;
	    	std::cout<<"^^^^^^^^^^^ ResultLeft: "<<left <<" "<<vertexX<<" "<<seg<<std::endl;
	    } while(loopCounter<=(seg/2));
	    std::cout<<"------------------ Point T1.15"<<std::endl;
	    minX = ((2.0*left)/seg-1) * IMSIZE * 0.3 + vertexX;

	    std::cout<<"^^^^^^^^^^^ Result: "<<v.GetX()<<" "<< minX <<" "<<minZ <<std::endl;
	    return true;
	}


	StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector, const float minX, const float minZ)
	{

		// // Get global TPC geometry information
		//   const LArTPCMap &larTPCMap(this->GetPandora().GetGeometry()->GetLArTPCMap());
		//   const LArTPC *const pFirstLArTPC(larTPCMap.begin()->second);

		//   const float minX(pFirstLArTPC->GetCenterX() - 0.5f * pFirstLArTPC->GetWidthX());
		//   const float widthX(pFirstLArTPC->GetWidthX());
		//   //const float minY(pFirstLArTPC->GetCenterY() - 0.5f * pFirstLArTPC->GetWidthY());
		//   //const float widthY(pFirstLArTPC->GetWidthY());
		//   const float minZ(pFirstLArTPC->GetCenterZ() - 0.5f * pFirstLArTPC->GetWidthZ());
		//   const float widthZ(pFirstLArTPC->GetWidthZ());

		std::ofstream file("OutTest/viewU.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

	const float hitNumber = 1.22f;//caloHitVector.size();
	file.write((char*)&hitNumber, sizeof(hitNumber));

	file.write((char*)&minX, sizeof(minX));
	file.write((char*)&minZ, sizeof(minZ));

	for (const CaloHit *const pCaloHit : caloHitVector)
	{
		std::array<float, 4> pixel = {0};

    	const float x = pCaloHit->GetPositionVector().GetX(); //(pCaloHit->GetPositionVector().GetX()-minX)/widthX; // Pixel number in X direction
    	const float z = pCaloHit->GetPositionVector().GetZ(); //(pCaloHit->GetPositionVector().GetZ()-minZ)/widthZ; // Pixel number in Z direction
    	file.write((char*)&x, sizeof(x));
    	file.write((char*)&z, sizeof(z));

    	const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());

		// Populates input image
    	pixel[0] = pCaloHit->GetHadronicEnergy();

    	// Populates prediction image
    	for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
    	{
    		const int particleID = mapEntry.first->GetParticleId();
    		switch(particleID){
    			case 22: case 11: case -11:
    			pixel[1] += mapEntry.second;
    			break;
    			case 2212:
    			pixel[2] += mapEntry.second;
    			break;
    		}
    	}
    	pixel[3] = 1.0f - pixel[1] - pixel[2];
    	file.write((char*)&pixel, sizeof(pixel));
    }
    file.close();
    return STATUS_CODE_SUCCESS;
}	

//------------------------------------------------------------------------------------------------------------------------------------------
StatusCode TrainingExportAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
// Read settings from xml file here
//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameU", m_caloHitListNameU));
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameV", m_caloHitListNameV));
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameW", m_caloHitListNameW));

	return STATUS_CODE_SUCCESS;
}
} // namespace lar_content
