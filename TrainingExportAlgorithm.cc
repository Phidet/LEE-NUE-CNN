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

#include <fstream>

using namespace pandora;
namespace lar_content
{
StatusCode TrainingExportAlgorithm::Run()
{
std::cout<<"Before -1 ______________"<<std::endl;
// ########## Taken from in parts from EventValidationAlgorithm.cc
// This generates the interaction type (6 for CCQE E, 7 for CCQE E+P) from the reconstructable (!) particles

	const MCParticleList *pMCParticleList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pMCParticleList));

    const CaloHitList *pCaloHitList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pCaloHitList));

    const PfoList *pPfoList(nullptr);
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));

	ValidationInfo validationInfo; //From EventValidationBaseAlgorithm
	FillValidationInfo(pMCParticleList, pCaloHitList, pPfoList, validationInfo); //From EventValidationAlgorithm

	//const LArMCParticleHelper::MCParticleToPfoHitSharingMap &mcToPfoHitSharingMap(validationInfo.GetInterpretedMCToPfoHitSharingMap());
	MCParticleList associatedMCPrimaries;

	MCParticleVector mcPrimaryVector;
    LArMonitoringHelper::GetOrderedMCParticleVector({validationInfo.GetTargetMCParticleToHitsMap()}, mcPrimaryVector);

	if(mcPrimaryVector.empty()) // End this run when there are no target MCParticles to process
	return STATUS_CODE_SUCCESS;

	for (const MCParticle *const pMCPrimary : mcPrimaryVector)
    {
        const bool hasMatch(mcToPfoHitSharingMap.count(pMCPrimary) && !mcToPfoHitSharingMap.at(pMCPrimary).empty());
        const bool isTargetPrimary(validationInfo.GetTargetMCParticleToHitsMap().count(pMCPrimary));

        if (!isTargetPrimary && !hasMatch)
            continue;

        associatedMCPrimaries.push_back(pMCPrimary);
	}
	const LArInteractionTypeHelper::InteractionType interactionType(LArInteractionTypeHelper::GetInteractionType(associatedMCPrimaries));
	const int interactionTypeInt(static_cast<int>(interactionType));
	// 	########## End of interaction type section

	if(interactionTypeInt!=6 && interactionTypeInt!=7) 
	return STATUS_CODE_SUCCESS;

	const CaloHitList *pCaloHitListU(nullptr);
	const CaloHitList *pCaloHitListV(nullptr);
	const CaloHitList *pCaloHitListW(nullptr);
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameV, pCaloHitListV));
	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameW, pCaloHitListW));

	const int entriesU = pCaloHitListU->size();
	const int entriesV = pCaloHitListV->size();
	const int entriesW = pCaloHitListW->size();

	CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
	CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
	CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());
	//std::sort(caloHitVectorU.begin(), caloHitVectorU.end(), LArClusterHelper::SortHitsByPosition);
	
	int protonHits(0);
	std::ostringstream tempStr; // Temporary storage for the data in case not enough proton hits are found and the event is dismissed 
	for (const CaloHit *const pCaloHit : caloHitVectorU)
	{
		if(MCParticleHelper::GetMainMCParticle(pCaloHit)->GetParticleId() == 2212)
			protonHits++;

		tempStr << "," << pCaloHit->GetPositionVector().GetX() << "," << pCaloHit->GetPositionVector().GetZ();
	}

	for(int i=0; i<2*(800-entriesU); i++) // Pads with zeros
		tempStr << ",0.0";	
// ---------------------------------
	for (const CaloHit *const pCaloHit : caloHitVectorV)
	{
		if(MCParticleHelper::GetMainMCParticle(pCaloHit)->GetParticleId() == 2212)
			protonHits++;

		tempStr << "," << pCaloHit->GetPositionVector().GetX() << "," << pCaloHit->GetPositionVector().GetZ();
	}

	for(int i=0; i<2*(800-entriesV); i++) // Pads with zeros
		tempStr << ",0.0";
// ---------------------------------
	for (const CaloHit *const pCaloHit : caloHitVectorW)
	{
		if(MCParticleHelper::GetMainMCParticle(pCaloHit)->GetParticleId() == 2212)
			protonHits++;

		tempStr << "," << pCaloHit->GetPositionVector().GetX() << "," << pCaloHit->GetPositionVector().GetZ();
	}

	for(int i=0; i<2*(800-entriesW); i++) // Pads with zeros
		tempStr << ",0.0";
// ----------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------
	std::ofstream myfile;
	myfile.open ("TrainingTestingOutput/TraTes.txt", std::ios::out | std::ios::app); // Appends all output to file   
	myfile << interactionType-6 <<","<< protonHits; // 0 for shower only and 1 for shower + 1 proton track
	myfile 	<< tempStr.str() << "\n";
	myfile.close();
	}

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

return EventValidationAlgorithm::ReadSettings(xmlHandle);
}
} // namespace lar_content
